/**
 * RL session
 *
 * @param {RLManager} manager
 * @param {String} id
 * @param {String} mode
 * @param {Room} room
 * @param {Object} options
 */
function RLSession(manager, id, mode, room, options)
{
    this.manager = manager;
    this.id = id;
    this.mode = mode;
    this.room = room;
    this.seed = typeof(options.seed) !== 'undefined' ? options.seed : null;
    this.grid = {
        width: options.gridWidth,
        height: options.gridHeight
    };
    this.actionRepeat = options.actionRepeat;
    this.fixedStep = options.fixedStep;
    this.warmupMs = options.warmupMs;
    this.warmdownMs = options.warmdownMs;
    this.printDelayMs = options.printDelayMs;
    this.mapSize = options.mapSize;
    this.autoAdvance = typeof(options.autoAdvance) !== 'undefined' ? options.autoAdvance : true;
    this.actors = new Collection([], 'id');
    this.done = false;
    this.lastResult = {
        round_winner_player_id: null,
        alive_player_ids: []
    };
    this.lastSnapshot = null;
    this.terminalSnapshot = null;
    this.lastTick = 0;
    this.currentGame = null;
    this.autoAdvanceTimer = null;
    this.lastActivityAt = Date.now();

    this.wsClients = [];

    this.onRoundEnd = this.onRoundEnd.bind(this);
    this.onGameEnd = this.onGameEnd.bind(this);
}

/**
 * Broadcast current state to all connected WebSocket clients
 */
RLSession.prototype.broadcastState = function()
{
    var message = JSON.stringify({ type: "state", data: this.buildState() });

    for (var i = this.wsClients.length - 1; i >= 0; i--) {
        try {
            this.wsClients[i].send(message);
        } catch (e) {
            this.wsClients.splice(i, 1);
        }
    }
};

/**
 * Update the last activity timestamp
 */
RLSession.prototype.touch = function()
{
    this.lastActivityAt = Date.now();
};

/**
 * Create and attach a bot actor
 *
 * @param {Object} data
 *
 * @return {Object}
 */
RLSession.prototype.addBot = function(data)
{
    this.touch();

    var client = new BotClient(data.name),
        player = null,
        actor;

    this.room.controller.attach(client, function () {});
    this.room.controller.onPlayerAdd(client, {
        name: data.name,
        color: data.color
    }, function (result) {
        if (result.success) {
            player = client.players.getFirst();
        }
    });

    if (!player) {
        this.room.controller.detach(client);

        return null;
    }

    actor = {
        id: player.id,
        client: client,
        player: player,
        controller: new BotController(player)
    };

    this.actors.add(actor);

    return actor;
};

/**
 * Find an actor by player id
 *
 * @param {String|Number} actorId
 *
 * @return {Object|null}
 */
RLSession.prototype.getActor = function(actorId)
{
    actorId = parseInt(actorId, 10);

    return this.actors.getById(actorId);
};

/**
 * Start a new episode
 *
 * @return {Object}
 */
RLSession.prototype.startEpisode = function()
{
    this.touch();

    if (this.room.game) {
        return this.buildState();
    }

    if (!this.room.manualGame && !this.room.isReady()) {
        return this.buildState();
    }

    this.done = false;
    this.lastResult.round_winner_player_id = null;
    this.lastResult.alive_player_ids = [];
    this.terminalSnapshot = null;
    this.lastSnapshot = null;
    this.lastTick = 0;

    if (this.room.manualGame) {
        this.room.randomGenerator = new SeededRandom(this.seed);
    }

    this.room.newGame();
    this.attachGame(this.room.game);

    if (typeof(this.mapSize) === 'number' && this.mapSize > 0) {
        this.room.game.size = this.mapSize;
        this.room.game.setSize();
        this.room.game.size = this.mapSize;
    }

    if (typeof(this.warmupMs) === 'number') {
        // If there are human players in the room (more players than bot actors),
        // enforce at least the default warmup so the human has time to orient
        var hasHumans = this.room.players.count() > this.actors.items.length;
        if (hasHumans && this.warmupMs < BaseGame.prototype.warmupTime) {
            this.room.game.warmupTime = BaseGame.prototype.warmupTime;
        } else {
            this.room.game.warmupTime = this.warmupMs;
        }
    }

    if (typeof(this.warmdownMs) === 'number') {
        this.room.game.warmdownTime = this.warmdownMs;
    }

    if (typeof(this.printDelayMs) === 'number') {
        this.room.game.printDelay = this.printDelayMs;
    }

    for (var i = 0; i < this.actors.items.length; i++) {
        this.actors.items[i].controller.setAction('straight');
        this.actors.items[i].client.emit('ready');
    }

    if (this.autoAdvance) {
        this.startAutoAdvance();
    }

    return this.buildState();
};

/**
 * Start auto-advancing the game at a fixed framerate
 */
RLSession.prototype.startAutoAdvance = function()
{
    this.stopAutoAdvance();

    var self = this;
    var interval = this.fixedStep || 16;

    this.autoAdvanceTimer = setInterval(function () {
        if (!self.room.game || !self.room.game.manual || self.done) {
            self.stopAutoAdvance();

            return;
        }

        self.room.game.advance(1);
        self.broadcastState();

        if (self.done) {
            self.broadcastState();
            self.stopAutoAdvance();
            self.drainGame();
        }
    }, interval);
};

/**
 * Stop auto-advancing
 */
RLSession.prototype.stopAutoAdvance = function()
{
    if (this.autoAdvanceTimer) {
        clearInterval(this.autoAdvanceTimer);
        this.autoAdvanceTimer = null;
    }
};

/**
 * Attach listeners to the active game
 *
 * @param {Game} game
 */
RLSession.prototype.attachGame = function(game)
{
    if (!game) {
        return;
    }

    this.detachGame();

    this.currentGame = game;
    game.on('round:end', this.onRoundEnd);
    game.on('end', this.onGameEnd);
};

/**
 * Detach listeners from the active game
 */
RLSession.prototype.detachGame = function()
{
    if (this.currentGame) {
        this.currentGame.removeListener('round:end', this.onRoundEnd);
        this.currentGame.removeListener('end', this.onGameEnd);
        this.currentGame = null;
    }
};

/**
 * Step a manual training session
 *
 * @param {Object} actions
 *
 * @return {Object}
 */
RLSession.prototype.step = function(actions)
{
    this.touch();

    if (!this.room.game || !this.room.game.manual) {
        return this.buildState();
    }

    // Pause auto-advance while stepping synchronously
    this.stopAutoAdvance();

    actions = actions || {};

    for (var i = 0; i < this.actors.items.length; i++) {
        var actor = this.actors.items[i],
            action = this.resolveActorAction(actor, actions);

        actor.controller.setAction(action);
    }

    this.room.game.advance(this.actionRepeat);

    if (this.done) {
        this.drainGame();

        return this.terminalSnapshot ? this.terminalSnapshot : this.buildState();
    }

    // Resume auto-advance for spectators
    if (this.autoAdvance) {
        this.startAutoAdvance();
    }

    return this.buildState();
};

/**
 * Resolve action for a given actor from a step payload
 *
 * @param {Object} actor
 * @param {Object} actions
 *
 * @return {String}
 */
RLSession.prototype.resolveActorAction = function(actor, actions)
{
    if (typeof(actions[actor.id]) !== 'undefined') {
        return actions[actor.id];
    }

    if (typeof(actions[actor.player.id]) !== 'undefined') {
        return actions[actor.player.id];
    }

    return actor.controller.action || 'straight';
};

/**
 * Set an action for a live or training actor
 *
 * @param {String|Number} actorId
 * @param {String} action
 *
 * @return {String|null}
 */
RLSession.prototype.setAction = function(actorId, action)
{
    this.touch();

    var actor = this.getActor(actorId);

    return actor ? actor.controller.setAction(action) : null;
};

/**
 * Hold an action for a number of ticks
 *
 * @param {String|Number} actorId
 * @param {String} action
 * @param {Number} ticks
 *
 * @return {String|null}
 */
RLSession.prototype.setAndHold = function(actorId, action, ticks)
{
    this.touch();

    var actor = this.getActor(actorId);

    return actor ? actor.controller.setAndHold(action, ticks) : null;
};

/**
 * Toggle an actor into the ready state if needed
 *
 * @param {String|Number} actorId
 *
 * @return {Boolean}
 */
RLSession.prototype.markReady = function(actorId)
{
    var actor = this.getActor(actorId),
        ready = false;

    if (!actor || actor.player.ready) {
        return actor ? actor.player.ready : false;
    }

    this.room.controller.onReady(actor.client, {player: actor.player.id}, function (result) {
        ready = result.success && result.ready;
    });

    return ready;
};

/**
 * Reset a training session to a fresh episode
 *
 * @return {Object}
 */
RLSession.prototype.reset = function()
{
    this.touch();
    this.drainGame();

    return this.startEpisode();
};

/**
 * Drain a finished manual game until it closes cleanly
 */
RLSession.prototype.drainGame = function()
{
    var guard = 4000;

    while (this.room.game && this.room.game.manual && guard-- > 0) {
        this.room.game.advance(1);
    }
};

/**
 * Get the current or terminal state
 *
 * @param {Object} options
 *
 * @return {Object}
 */
RLSession.prototype.buildState = function(options)
{
    if (!this.room.game) {
        return this.terminalSnapshot || this.lastSnapshot || this.manager.stateBuilder.build(this, options);
    }

    this.lastSnapshot = this.manager.stateBuilder.build(this, options);
    this.lastTick = this.room.game.tick;

    return this.lastSnapshot;
};

/**
 * Close the session and remove all bot actors
 */
RLSession.prototype.close = function()
{
    this.stopAutoAdvance();
    this.detachGame();

    for (var i = this.actors.items.length - 1; i >= 0; i--) {
        this.room.controller.detach(this.actors.items[i].client);
    }

    this.actors.clear();

    for (var j = 0; j < this.wsClients.length; j++) {
        try {
            this.wsClients[j].close();
        } catch (e) {}
    }
    this.wsClients = [];

    // Remove training room from the repository
    if (this.mode === 'training') {
        this.manager.server.roomRepository.remove(this.room);
    }
};

/**
 * On round end
 *
 * @param {Object} data
 */
RLSession.prototype.onRoundEnd = function(data)
{
    this.done = true;
    this.stopAutoAdvance();
    this.lastResult.round_winner_player_id = data.winner ? data.winner.id : null;
    this.lastResult.alive_player_ids = data.winner ? [data.winner.id] : [];
    this.terminalSnapshot = this.buildState();
    this.broadcastState();
};

/**
 * On game end
 *
 * @param {Object} data
 */
RLSession.prototype.onGameEnd = function(data)
{
    this.lastTick = data.game.tick;
    this.detachGame();
};
