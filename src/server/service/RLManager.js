/**
 * RL manager
 *
 * @param {Server} server
 */
/**
 * Default session timeout: 5 minutes of inactivity
 *
 * @type {Number}
 */
RLManager.sessionTimeout = 5 * 60 * 1000;

/**
 * Cleanup sweep interval: every 60 seconds
 *
 * @type {Number}
 */
RLManager.cleanupInterval = 60 * 1000;

function RLManager(server)
{
    this.server = server;
    this.sessions = new Collection([], 'id');
    this.stateBuilder = new RLStateBuilder();
    this.sessionId = 0;

    this.cleanup = this.cleanup.bind(this);
    this.cleanupTimer = setInterval(this.cleanup, RLManager.cleanupInterval);
}

/**
 * Create a new RL session
 *
 * @param {Object} data
 *
 * @return {RLSession|null}
 */
RLManager.prototype.createSession = function(data)
{
    data = data || {};

    var mode = data.mode === 'live' ? 'live' : 'training',
        room = mode === 'live' ? this.server.roomRepository.get(data.room_name) : this.createTrainingRoom(data),
        session;

    if (!room) {
        return null;
    }

    session = new RLSession(this, this.nextSessionId(), mode, room, {
        seed: typeof(data.seed) !== 'undefined' ? data.seed : null,
        gridWidth: this.getGridSize(data.grid_width),
        gridHeight: this.getGridSize(data.grid_height),
        actionRepeat: this.getActionRepeat(data.action_repeat),
        fixedStep: this.getFixedStep(data.fixed_step),
        warmupMs: this.getDelay(data.warmup_ms),
        warmdownMs: this.getDelay(data.warmdown_ms),
        printDelayMs: this.getDelay(data.print_delay_ms),
        mapSize: typeof(data.map_size) === 'number' ? data.map_size : undefined,
        autoAdvance: data.auto_advance !== false
    });

    this.sessions.add(session);

    return session;
};

/**
 * Create a hidden manual training room
 *
 * @param {Object} data
 *
 * @return {Room}
 */
RLManager.prototype.createTrainingRoom = function(data)
{
    var room = new Room('rl-session-' + (this.sessionId + 1));

    room.manualGame = true;
    room.fixedStep = this.getFixedStep(data.fixed_step);
    room.randomGenerator = new SeededRandom(typeof(data.seed) !== 'undefined' ? data.seed : room.name);
    room.config.setOpen(true);
    room.config.setMaxScore(typeof(data.max_score) !== 'undefined' ? data.max_score : 1);

    if (typeof(data.bonus_rate) !== 'undefined') {
        room.config.setVariable('bonusRate', data.bonus_rate);
    }

    this.applyBonuses(room, data.bonuses);

    // Register in the room repository so the frontend can spectate via WebSocket
    this.server.roomRepository.rooms.add(room);
    room.on('close', this.server.roomRepository.onRoomClose);

    return room;
};

/**
 * Apply a bonus configuration to a room
 *
 * @param {Room} room
 * @param {Object} bonuses
 */
RLManager.prototype.applyBonuses = function(room, bonuses)
{
    if (!bonuses) {
        return;
    }

    for (var bonus in bonuses) {
        if (bonuses.hasOwnProperty(bonus) && room.config.bonusExists(bonus)) {
            room.config.setBonus(bonus, bonuses[bonus]);
        }
    }
};

/**
 * Find a session by seed
 *
 * @param {String} seed
 *
 * @return {RLSession|null}
 */
RLManager.prototype.findSessionBySeed = function(seed)
{
    for (var i = 0; i < this.sessions.items.length; i++) {
        if (this.sessions.items[i].seed === seed) {
            return this.sessions.items[i];
        }
    }

    return null;
};

/**
 * Get a session
 *
 * @param {String} id
 *
 * @return {RLSession|null}
 */
RLManager.prototype.getSession = function(id)
{
    return this.sessions.getById(id);
};

/**
 * Remove a session
 *
 * @param {RLSession} session
 */
RLManager.prototype.removeSession = function(session)
{
    if (session && this.sessions.remove(session)) {
        session.close();
    }
};

/**
 * Clamp grid size
 *
 * @param {Number} value
 *
 * @return {Number}
 */
RLManager.prototype.getGridSize = function(value)
{
    value = parseInt(value, 10);

    if (!value || value < 8) {
        return 100;
    }

    return Math.min(128, value);
};

/**
 * Clamp action repeat
 *
 * @param {Number} value
 *
 * @return {Number}
 */
RLManager.prototype.getActionRepeat = function(value)
{
    value = parseInt(value, 10);

    if (!value || value < 1) {
        return 4;
    }

    return Math.min(32, value);
};

/**
 * Clamp fixed step duration
 *
 * @param {Number} value
 *
 * @return {Number}
 */
RLManager.prototype.getFixedStep = function(value)
{
    value = parseFloat(value);

    if (!value || value <= 0) {
        return BaseGame.prototype.framerate;
    }

    return value;
};

/**
 * Normalize a delay override
 *
 * @param {Number} value
 *
 * @return {Number|null}
 */
RLManager.prototype.getDelay = function(value)
{
    if (typeof(value) === 'undefined' || value === null || value === '') {
        return null;
    }

    value = parseInt(value, 10);

    return value >= 0 ? value : null;
};

/**
 * Generate a stable session id
 *
 * @return {String}
 */
RLManager.prototype.nextSessionId = function()
{
    this.sessionId++;

    return 'rl:' + this.sessionId;
};

/**
 * Remove sessions that have been inactive for longer than the timeout
 */
RLManager.prototype.cleanup = function()
{
    var now = Date.now(),
        timeout = RLManager.sessionTimeout,
        stale = [];

    for (var i = 0; i < this.sessions.items.length; i++) {
        if (now - this.sessions.items[i].lastActivityAt > timeout) {
            stale.push(this.sessions.items[i]);
        }
    }

    for (var j = 0; j < stale.length; j++) {
        console.log('Cleaning up inactive RL session %s (idle %ds)', stale[j].id, Math.round((now - stale[j].lastActivityAt) / 1000));
        this.removeSession(stale[j]);
    }
};
