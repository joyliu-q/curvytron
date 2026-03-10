/**
 * Mock Socket Client for RL agents (no real WebSocket)
 */
function MockSocketClient(id)
{
    EventEmitter.call(this);

    this.id        = id;
    this.active    = true;
    this.connected = true;
    this.players   = new Collection([], 'id');
    this.events    = [];
    this.loop      = null;

    this.pingLogger = { start: function(){}, stop: function(){} };
    this.tickrate   = { tick: function(){}, stop: function(){} };
}

MockSocketClient.prototype = Object.create(EventEmitter.prototype);
MockSocketClient.prototype.constructor = MockSocketClient;

MockSocketClient.prototype.addEvent    = function() {};
MockSocketClient.prototype.addEvents   = function() {};
MockSocketClient.prototype.sendEvents  = function() {};
MockSocketClient.prototype.flush       = function() {};
MockSocketClient.prototype.start       = function() {};
MockSocketClient.prototype.stop        = function() {};
MockSocketClient.prototype.isPlaying   = function() { return !this.players.isEmpty(); };
MockSocketClient.prototype.clearPlayers = function() { this.players.clear(); };
MockSocketClient.prototype.serialize   = function() { return { id: this.id, active: this.active }; };

/**
 * RL Environment - wraps a game for stepped RL control
 *
 * @param {String} name
 * @param {Number} numPlayers
 * @param {Object} options
 */
function RLEnvironment(name, numPlayers, options)
{
    options = options || {};

    this.id              = name;
    this.numPlayers      = numPlayers;
    this.fixedStep       = options.fixedStep || (1/60 * 1000);
    this.stepsPerAction  = options.stepsPerAction || 4;
    this.noBonuses       = options.noBonuses || false;
    this.stepCount       = 0;
    this.totalSteps      = 0;
    this.roundNumber     = 0;
    this.roundDone       = false;
    this.gameDone        = false;
    this.previousAlive   = {};
    this.colors          = ['#FF0000', '#00FF00', '#0044FF', '#FFFF00', '#FF00FF', '#00FFFF', '#FF8800', '#88FF00'];

    this.room        = new Room(name);
    this.mockClients = [];
    this.agentIds    = [];

    for (var i = 0; i < numPlayers; i++) {
        var client = new MockSocketClient('rl-' + name + '-' + i);
        this.mockClients.push(client);
        this.room.controller.clients.add(client);

        var player = new Player(client, 'Agent-' + i, this.colors[i % this.colors.length]);
        player.toggleReady(true);
        this.room.addPlayer(player);
        client.players.add(player);
        this.agentIds.push(player.id);
    }

    if (this.noBonuses) {
        this.room.config.bonuses = {};
    }

    this.game = null;
}

/**
 * Reset the environment, start a new round
 *
 * @return {Object} initial observation
 */
RLEnvironment.prototype.reset = function()
{
    this.cleanup();

    for (var i = this.room.players.items.length - 1; i >= 0; i--) {
        var p = this.room.players.items[i];
        if (p.avatar) {
            p.avatar.destroy();
            p.avatar = null;
        }
        p.ready = true;
    }
    this.room.game = null;

    this.game = new Game(this.room);
    this.room.game = this.game;

    this.game.warmupTime   = 0;
    this.game.warmdownTime = 0;

    var env = this;
    var origEndRound = this.game.endRound.bind(this.game);
    this.game.endRound = function() {
        if (env.game.inRound) {
            env.game.inRound = false;
            env.game.onRoundEnd();
            env.roundDone = true;
        }
    };

    this.game.started = true;
    this.game.inRound = true;
    this.game.onRoundNew();

    this.game.rendered = new Date().getTime();
    this.game.bonusManager.start();
    this.game.world.activate();

    for (var j = this.game.avatars.items.length - 1; j >= 0; j--) {
        var avatar = this.game.avatars.items[j];
        avatar.ready = true;
        avatar.printManager.start();
    }

    this.stepCount     = 0;
    this.roundNumber++;
    this.roundDone     = false;
    this.gameDone      = false;
    this.previousAlive = {};

    for (var k = this.game.avatars.items.length - 1; k >= 0; k--) {
        this.previousAlive[this.game.avatars.items[k].id] = true;
    }

    return this.getObservation();
};

/**
 * Step the environment forward
 *
 * @param {Object} actions  - map of avatar id to move (-1, 0, 1)
 * @param {Number} steps    - number of physics frames to advance
 *
 * @return {Object} { observation, rewards, done, info }
 */
RLEnvironment.prototype.step = function(actions, steps)
{
    if (!this.game || this.roundDone) {
        return { error: 'Round is not active. Call reset first.' };
    }

    steps = steps || this.stepsPerAction;

    for (var id in actions) {
        if (actions.hasOwnProperty(id)) {
            var avatar = this.game.avatars.getById(parseInt(id, 10) || id);
            if (avatar && avatar.alive) {
                avatar.updateAngularVelocity(actions[id]);
            }
        }
    }

    var deathsThisStep = [];

    for (var s = 0; s < steps; s++) {
        if (this.roundDone) { break; }

        this.game.update(this.fixedStep);
        this.stepCount++;
        this.totalSteps++;

        for (var a = this.game.avatars.items.length - 1; a >= 0; a--) {
            var av = this.game.avatars.items[a];
            if (this.previousAlive[av.id] && !av.alive) {
                deathsThisStep.push(av.id);
                this.previousAlive[av.id] = false;
            }
        }
    }

    var rewards = this.computeRewards(deathsThisStep);

    var won = null;
    if (this.roundDone) {
        won = this.game.isWon();
        if (won) {
            this.gameDone = true;
        }
    }

    return {
        observation: this.getObservation(),
        rewards: rewards,
        done: this.roundDone,
        info: {
            step: this.stepCount,
            total_steps: this.totalSteps,
            round: this.roundNumber,
            deaths_this_step: deathsThisStep,
            round_winner: this.game.roundWinner ? this.game.roundWinner.id : null,
            game_over: this.gameDone,
            game_winner: (won && won instanceof Avatar) ? won.id : null
        }
    };
};

/**
 * Compute rewards for each agent
 */
RLEnvironment.prototype.computeRewards = function(deathsThisStep)
{
    var rewards = {};
    var aliveCount = 0;

    for (var i = this.game.avatars.items.length - 1; i >= 0; i--) {
        var avatar = this.game.avatars.items[i];
        if (avatar.alive) { aliveCount++; }
    }

    for (var j = this.game.avatars.items.length - 1; j >= 0; j--) {
        var av = this.game.avatars.items[j];
        var r = 0;

        if (av.alive) {
            r += 0.1;
        }

        if (deathsThisStep.indexOf(av.id) >= 0) {
            r -= 10;
        }

        if (this.roundDone && av.alive) {
            r += 25;
        }

        rewards[av.id] = r;
    }

    return rewards;
};

/**
 * Get full game state observation
 */
RLEnvironment.prototype.getObservation = function()
{
    if (!this.game) { return null; }

    var avatars = [];
    for (var i = 0; i < this.game.avatars.items.length; i++) {
        var av = this.game.avatars.items[i];
        avatars.push({
            id:               av.id,
            name:             av.name,
            x:                av.x,
            y:                av.y,
            angle:            av.angle,
            velocity:         av.velocity,
            angular_velocity: av.angularVelocity,
            radius:           av.radius,
            alive:            av.alive,
            invincible:       av.invincible,
            inverse:          av.inverse,
            printing:         av.printing,
            score:            av.score,
            round_score:      av.roundScore,
            trail_length:     av.trail.points.length
        });
    }

    var bonuses = [];
    if (this.game.bonusManager && this.game.bonusManager.bonuses) {
        for (var b = 0; b < this.game.bonusManager.bonuses.items.length; b++) {
            var bonus = this.game.bonusManager.bonuses.items[b];
            bonuses.push({
                id:   bonus.id,
                x:    bonus.x,
                y:    bonus.y,
                type: bonus.constructor.name
            });
        }
    }

    return {
        map_size:    this.game.size,
        borderless:  this.game.borderless,
        in_round:    this.game.inRound,
        step:        this.stepCount,
        round:       this.roundNumber,
        avatars:     avatars,
        bonuses:     bonuses
    };
};

/**
 * Get nearby trail bodies for a specific avatar (spatial awareness for RL)
 *
 * @param {Number|String} avatarId
 * @param {Number} lookAhead - distance ahead of the avatar to scan
 */
RLEnvironment.prototype.getNearbyTrails = function(avatarId, lookAhead)
{
    if (!this.game) { return []; }

    lookAhead = lookAhead || 20;

    var avatar = this.game.avatars.getById(parseInt(avatarId, 10) || avatarId);
    if (!avatar) { return []; }

    var bodies = [];
    var scanRadius = lookAhead;

    for (var i = this.game.world.islands.items.length - 1; i >= 0; i--) {
        var island = this.game.world.islands.items[i];
        var dx = (island.fromX + island.size / 2) - avatar.x;
        var dy = (island.fromY + island.size / 2) - avatar.y;
        var dist = Math.sqrt(dx * dx + dy * dy);

        if (dist > scanRadius + island.size) { continue; }

        for (var j = island.bodies.items.length - 1; j >= 0; j--) {
            var body = island.bodies.items[j];
            var bx = body.x - avatar.x;
            var by = body.y - avatar.y;
            if (bx * bx + by * by < scanRadius * scanRadius) {
                bodies.push({
                    x:      body.x,
                    y:      body.y,
                    radius: body.radius,
                    owner:  (body.data && body.data.id) ? body.data.id : null
                });
            }
        }
    }

    return bodies;
};

/**
 * Get wall distances for an avatar (distance to each border)
 */
RLEnvironment.prototype.getWallDistances = function(avatarId)
{
    if (!this.game) { return null; }

    var avatar = this.game.avatars.getById(parseInt(avatarId, 10) || avatarId);
    if (!avatar) { return null; }

    var size = this.game.size;
    return {
        left:   avatar.x,
        right:  size - avatar.x,
        top:    avatar.y,
        bottom: size - avatar.y
    };
};

/**
 * Clean up timers and game state
 */
RLEnvironment.prototype.cleanup = function()
{
    if (this.game) {
        if (this.game.frame) {
            this.game.clearFrame();
        }
        if (this.game.bonusManager) {
            this.game.bonusManager.stop();
        }
        if (this.game.controller && this.game.controller.waiting) {
            clearTimeout(this.game.controller.waiting);
            this.game.controller.waiting = null;
        }
    }
};

/**
 * Destroy the environment
 */
RLEnvironment.prototype.destroy = function()
{
    this.cleanup();

    if (this.game && this.game.world) {
        this.game.world.clear();
    }

    this.game = null;
    this.room.game = null;
};

/**
 * RL Controller - Express routes for the RL API
 *
 * @param {Server} server
 */
function RLController(server)
{
    this.server = server;
    this.envs   = {};

    this.mountRoutes();
}

/**
 * Mount Express routes
 */
RLController.prototype.mountRoutes = function()
{
    var self = this;
    var app  = this.server.app;

    app.get('/api/rl/info', function(req, res) {
        res.json({
            name: 'curvytron-rl',
            version: '1.0.0',
            action_space: { type: 'discrete', values: [-1, 0, 1], description: 'left, straight, right' },
            defaults: {
                fixed_step_ms: 1/60 * 1000,
                steps_per_action: 4,
                avatar_velocity: BaseAvatar.prototype.velocity,
                avatar_angular_velocity_base: BaseAvatar.prototype.angularVelocityBase,
                avatar_radius: BaseAvatar.prototype.radius,
                framerate_ms: BaseGame.prototype.framerate
            }
        });
    });

    app.get('/api/rl/envs', function(req, res) {
        var envList = [];
        for (var id in self.envs) {
            if (self.envs.hasOwnProperty(id)) {
                var env = self.envs[id];
                envList.push({
                    id: env.id,
                    num_players: env.numPlayers,
                    step: env.stepCount,
                    total_steps: env.totalSteps,
                    round: env.roundNumber,
                    round_done: env.roundDone,
                    game_done: env.gameDone
                });
            }
        }
        res.json({ environments: envList });
    });

    app.post('/api/rl/create', function(req, res) {
        var name       = req.body.name || 'rl-' + Date.now();
        var numPlayers = req.body.num_players || 2;
        var options    = req.body.options || {};

        if (self.envs[name]) {
            return res.status(409).json({ error: 'Environment "' + name + '" already exists.' });
        }

        if (numPlayers < 1 || numPlayers > 8) {
            return res.status(400).json({ error: 'num_players must be between 1 and 8.' });
        }

        var env = new RLEnvironment(name, numPlayers, options);
        self.envs[name] = env;

        res.json({
            id: name,
            num_players: numPlayers,
            agent_ids: env.agentIds,
            options: {
                fixed_step: env.fixedStep,
                steps_per_action: env.stepsPerAction,
                no_bonuses: env.noBonuses
            }
        });
    });

    app.post('/api/rl/reset/:envId', function(req, res) {
        var env = self.envs[req.params.envId];
        if (!env) {
            return res.status(404).json({ error: 'Environment not found.' });
        }

        var observation = env.reset();
        res.json({ observation: observation });
    });

    app.post('/api/rl/step/:envId', function(req, res) {
        var env = self.envs[req.params.envId];
        if (!env) {
            return res.status(404).json({ error: 'Environment not found.' });
        }

        var actions = req.body.actions || {};
        var steps   = req.body.steps;

        var result = env.step(actions, steps);
        res.json(result);
    });

    app.get('/api/rl/state/:envId', function(req, res) {
        var env = self.envs[req.params.envId];
        if (!env) {
            return res.status(404).json({ error: 'Environment not found.' });
        }

        var observation = env.getObservation();
        res.json({ observation: observation });
    });

    app.get('/api/rl/trails/:envId/:avatarId', function(req, res) {
        var env = self.envs[req.params.envId];
        if (!env) {
            return res.status(404).json({ error: 'Environment not found.' });
        }

        var lookAhead = parseInt(req.query.radius, 10) || 20;
        var trails    = env.getNearbyTrails(req.params.avatarId, lookAhead);
        var walls     = env.getWallDistances(req.params.avatarId);

        res.json({ trails: trails, walls: walls });
    });

    app.delete('/api/rl/close/:envId', function(req, res) {
        var env = self.envs[req.params.envId];
        if (!env) {
            return res.status(404).json({ error: 'Environment not found.' });
        }

        env.destroy();
        delete self.envs[req.params.envId];

        res.json({ success: true });
    });
};
