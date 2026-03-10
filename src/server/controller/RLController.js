/**
 * RL API controller
 *
 * @param {Server} server
 */
function RLController(server)
{
    this.server = server;
    this.manager = new RLManager(server);

    this.createSession = this.createSession.bind(this);
    this.getSessionState = this.getSessionState.bind(this);
    this.addBot = this.addBot.bind(this);
    this.startSession = this.startSession.bind(this);
    this.resetSession = this.resetSession.bind(this);
    this.stepSession = this.stepSession.bind(this);
    this.setActorAction = this.setActorAction.bind(this);
    this.setAndHoldAction = this.setAndHoldAction.bind(this);
    this.readyActor = this.readyActor.bind(this);
    this.deleteSession = this.deleteSession.bind(this);

    this.attachRoutes();
}

/**
 * Attach API routes
 */
RLController.prototype.attachRoutes = function()
{
    this.server.app.get('/api/rl/sessions', this.listSessions);
    this.server.app.post('/api/rl/sessions', this.withJsonBody(this.createSession));
    this.server.app.get('/api/rl/sessions/:sessionId/state', this.getSessionState);
    this.server.app.post('/api/rl/sessions/:sessionId/bots', this.withJsonBody(this.addBot));
    this.server.app.post('/api/rl/sessions/:sessionId/start', this.startSession);
    this.server.app.post('/api/rl/sessions/:sessionId/reset', this.resetSession);
    this.server.app.post('/api/rl/sessions/:sessionId/step', this.withJsonBody(this.stepSession));
    this.server.app.post('/api/rl/sessions/:sessionId/actors/:actorId/action', this.withJsonBody(this.setActorAction));
    this.server.app.post('/api/rl/sessions/:sessionId/actors/:actorId/set-and-hold', this.withJsonBody(this.setAndHoldAction));
    this.server.app.post('/api/rl/sessions/:sessionId/actors/:actorId/ready', this.readyActor);
    this.server.app.delete('/api/rl/sessions/:sessionId', this.deleteSession);
};

/**
 * Wrap a handler with JSON body parsing
 *
 * @param {Function} handler
 *
 * @return {Function}
 */
RLController.prototype.withJsonBody = function(handler)
{
    return function (req, res) {
        if (typeof(req.body) !== 'undefined' && req.body !== null) {
            return handler(req, res);
        }

        var raw = '';

        req.on('data', function (chunk) { raw += chunk; });
        req.on('end', function () {
            if (!raw.length) {
                req.body = {};

                return handler(req, res);
            }

            try {
                req.body = JSON.parse(raw);
            } catch (error) {
                return this.sendJson(res, 400, {error: 'Invalid JSON body'});
            }

            return handler(req, res);
        }.bind(this));
    }.bind(this);
};

/**
 * Find a session or respond with 404
 *
 * @param {Object} req
 * @param {Object} res
 *
 * @return {RLSession|null}
 */
RLController.prototype.requireSession = function(req, res)
{
    var session = this.manager.getSession(req.params.sessionId);

    if (!session) {
        this.sendJson(res, 404, {error: 'Unknown session'});

        return null;
    }

    return session;
};

/**
 * List sessions, optionally filtered by seed
 *
 * @param {Object} req
 * @param {Object} res
 */
RLController.prototype.listSessions = function(req, res)
{
    var self = this;
    var sessions = this.manager.sessions.items.map(function(session) {
        return self.serializeSession(session);
    });

    var seed = req.query && req.query.seed;

    if (seed) {
        sessions = sessions.filter(function(s) { return s.seed === seed; });
    }

    this.sendJson(res, 200, sessions);
};

/**
 * Create a session
 *
 * @param {Object} req
 * @param {Object} res
 */
RLController.prototype.createSession = function(req, res)
{
    // If a session with this seed already exists, return it
    if (req.body.seed) {
        var existing = this.manager.findSessionBySeed(req.body.seed);

        if (existing) {
            return this.sendJson(res, 200, this.serializeSession(existing));
        }
    }

    var session = this.manager.createSession(req.body);

    if (!session) {
        return this.sendJson(res, 404, {error: 'Unable to create session'});
    }

    if (req.body.bots instanceof Array) {
        for (var i = 0; i < req.body.bots.length; i++) {
            session.addBot(req.body.bots[i]);
        }
    }

    this.sendJson(res, 201, this.serializeSession(session));
};

/**
 * Get current state for a session
 *
 * @param {Object} req
 * @param {Object} res
 */
RLController.prototype.getSessionState = function(req, res)
{
    var session = this.requireSession(req, res);

    if (!session) {
        return;
    }

    this.sendJson(res, 200, session.buildState({
        gridWidth: req.query.grid_width,
        gridHeight: req.query.grid_height
    }));
};

/**
 * Add a bot actor to a session
 *
 * @param {Object} req
 * @param {Object} res
 */
RLController.prototype.addBot = function(req, res)
{
    var session = this.requireSession(req, res),
        actor;

    if (!session) {
        return;
    }

    actor = session.addBot({
        name: req.body.name || ('bot-' + (session.actors.count() + 1)),
        color: req.body.color
    });

    if (!actor) {
        return this.sendJson(res, 400, {error: 'Unable to add bot'});
    }

    this.sendJson(res, 201, this.serializeActor(actor));
};

/**
 * Start a session episode
 *
 * @param {Object} req
 * @param {Object} res
 */
RLController.prototype.startSession = function(req, res)
{
    var session = this.requireSession(req, res);

    if (!session) {
        return;
    }

    this.sendJson(res, 200, session.startEpisode());
};

/**
 * Reset a session episode
 *
 * @param {Object} req
 * @param {Object} res
 */
RLController.prototype.resetSession = function(req, res)
{
    var session = this.requireSession(req, res);

    if (!session) {
        return;
    }

    this.sendJson(res, 200, session.reset());
};

/**
 * Step a session
 *
 * @param {Object} req
 * @param {Object} res
 */
RLController.prototype.stepSession = function(req, res)
{
    var session = this.requireSession(req, res);

    if (!session) {
        return;
    }

    this.sendJson(res, 200, session.step(req.body.actions));
};

/**
 * Set an actor action
 *
 * @param {Object} req
 * @param {Object} res
 */
RLController.prototype.setActorAction = function(req, res)
{
    var session = this.requireSession(req, res),
        action;

    if (!session) {
        return;
    }

    action = session.setAction(req.params.actorId, req.body.action);

    if (action === null) {
        return this.sendJson(res, 404, {error: 'Unknown actor'});
    }

    this.sendJson(res, 200, {action: action});
};

/**
 * Hold an actor action for a number of ticks
 *
 * @param {Object} req
 * @param {Object} res
 */
RLController.prototype.setAndHoldAction = function(req, res)
{
    var session = this.requireSession(req, res),
        action;

    if (!session) {
        return;
    }

    action = session.setAndHold(req.params.actorId, req.body.action, req.body.ticks);

    if (action === null) {
        return this.sendJson(res, 404, {error: 'Unknown actor'});
    }

    this.sendJson(res, 200, {action: action});
};

/**
 * Ready an actor before a live game starts
 *
 * @param {Object} req
 * @param {Object} res
 */
RLController.prototype.readyActor = function(req, res)
{
    var session = this.requireSession(req, res),
        ready;

    if (!session) {
        return;
    }

    ready = session.markReady(req.params.actorId);

    this.sendJson(res, 200, {ready: ready});
};

/**
 * Delete a session
 *
 * @param {Object} req
 * @param {Object} res
 */
RLController.prototype.deleteSession = function(req, res)
{
    var session = this.requireSession(req, res);

    if (!session) {
        return;
    }

    this.manager.removeSession(session);
    this.sendJson(res, 200, {success: true});
};

/**
 * Serialize session metadata
 *
 * @param {RLSession} session
 *
 * @return {Object}
 */
RLController.prototype.serializeSession = function(session)
{
    return {
        session_id: session.id,
        room_name: session.room.name,
        mode: session.mode,
        seed: session.seed,
        grid: session.grid,
        action_repeat: session.actionRepeat,
        fixed_step: session.fixedStep,
        warmup_ms: session.warmupMs,
        warmdown_ms: session.warmdownMs,
        print_delay_ms: session.printDelayMs,
        actors: session.actors.map(function () {
            return {
                id: this.id,
                player_id: this.player.id,
                client_id: this.client.id,
                name: this.player.name,
                color: this.player.color
            };
        }).items,
        state: session.buildState()
    };
};

/**
 * Serialize actor metadata
 *
 * @param {Object} actor
 *
 * @return {Object}
 */
RLController.prototype.serializeActor = function(actor)
{
    return {
        id: actor.id,
        player_id: actor.player.id,
        client_id: actor.client.id,
        name: actor.player.name,
        color: actor.player.color
    };
};

/**
 * Send a JSON response
 *
 * @param {Object} res
 * @param {Number} status
 * @param {Object} body
 */
RLController.prototype.sendJson = function(res, status, body)
{
    res.status(status);
    res.set('Content-Type', 'application/json');
    res.send(JSON.stringify(body));
};
