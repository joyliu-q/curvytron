/**
 * BaseGame
 *
 * @param {Room} room
 */
function BaseGame(room)
{
    EventEmitter.call(this);

    this.room         = room;
    this.name         = this.room.name;
    this.frame        = null;
    this.avatars      = this.room.players.map(function () { return this.getAvatar(); });
    this.size         = this.getSize(this.avatars.count());
    this.rendered     = null;
    this.maxScore     = room.config.getMaxScore();
    this.fps          = new FPSLogger();
    this.started      = false;
    this.bonusManager = new BonusManager(this, room.config.getBonuses(), room.config.getVariable('bonusRate'));
    this.inRound      = false;
    this.randomGenerator = room && room.randomGenerator ? room.randomGenerator : null;
    this.manual       = room && room.manualGame ? true : false;
    this.fixedStep    = typeof(room.fixedStep) === 'number' ? room.fixedStep : this.framerate;
    this.clock        = 0;
    this.tick         = 0;
    this.schedules    = [];
    this.scheduleId   = 0;

    this.start    = this.start.bind(this);
    this.stop     = this.stop.bind(this);
    this.loop     = this.loop.bind(this);
    this.newRound = this.newRound.bind(this);
    this.endRound = this.endRound.bind(this);
    this.end      = this.end.bind(this);
    this.onFrame  = this.onFrame.bind(this);
    this.advance  = this.advance.bind(this);
    this.random   = this.random.bind(this);
}

BaseGame.prototype = Object.create(EventEmitter.prototype);
BaseGame.prototype.constructor = BaseGame;

/**
 * Loop frame rate
 *
 * @type {Number}
 */
BaseGame.prototype.framerate = 1/60 * 1000;

/**
 * Random float from runtime RNG
 *
 * @return {Float}
 */
BaseGame.prototype.random = function()
{
    return this.randomGenerator ? this.randomGenerator.next() : Math.random();
};

/**
 * Map size factor per player
 *
 * @type {Number}
 */
BaseGame.prototype.perPlayerSize = 80;

/**
 * Time before round start
 *
 * @type {Number}
 */
BaseGame.prototype.warmupTime = 3000;

/**
 * Time after round end
 *
 * @type {Number}
 */
BaseGame.prototype.warmdownTime = 5000;

/**
 * Margin from borders
 *
 * @type {Number}
 */
BaseGame.prototype.spawnMargin = 0.05;

/**
 * Angle margin from borders
 *
 * @type {Number}
 */
BaseGame.prototype.spawnAngleMargin = 0.3;

/**
 * Borderless
 *
 * @type {Boolean}
 */
BaseGame.prototype.borderless = false;

/**
 * Update
 *
 * @param {Number} step
 */
BaseGame.prototype.update = function(step) {};

/**
 * Remove a avatar from the game
 *
 * @param {Avatar} avatar
 */
BaseGame.prototype.removeAvatar = function(avatar)
{
    if (this.avatars.exists(avatar)) {
        avatar.die();
        avatar.destroy();
    }
};

/**
 * Start loop
 */
BaseGame.prototype.start = function()
{
    if (!this.frame) {
        if (this.manual) {
            this.frame = true;
            this.onStart();

            return;
        }

        this.onStart();
        this.loop();
    }
};

/**
 * Stop loop
 */
BaseGame.prototype.stop = function()
{
    if (this.frame) {
        if (this.manual) {
            this.frame = null;
            this.onStop();

            return;
        }

        this.clearFrame();
        this.onStop();
    }
};

/**
 * Animation loop
 */
BaseGame.prototype.loop = function()
{
    this.newFrame();

    var now  = new Date().getTime(),
        step = now - this.rendered;

    this.rendered = now;

    this.onFrame(step);
    this.fps.onFrame();
};

/**
 * On start
 */
BaseGame.prototype.onStart = function()
{
    this.rendered = this.manual ? this.clock : new Date().getTime();
    this.bonusManager.start();
    this.fps.start();
};

/**
 * Onn stop
 */
BaseGame.prototype.onStop = function()
{
    this.rendered = null;
    this.bonusManager.stop();
    this.fps.stop();

    var size = this.getSize(this.getPresentAvatars().count());

    if (this.size !== size) {
        this.setSize(size);
    }
};

/**
 * On round new
 */
BaseGame.prototype.onRoundNew = function()
{
    this.borderless = BaseGame.prototype.borderless;
    this.tick = 0;

    this.bonusManager.clear();

    for (var i = this.avatars.items.length - 1; i >= 0; i--) {
        if (this.avatars.items[i].present) {
            this.avatars.items[i].clear();
        }
    }
};

/**
 * On round end
 */
BaseGame.prototype.onRoundEnd = function() {};

/**
 * Get new frame
 */
BaseGame.prototype.newFrame = function()
{
    if (this.manual) {
        return;
    }

    this.frame = setTimeout(this.loop, this.framerate);
};

/**
 * Clear frame
 */
BaseGame.prototype.clearFrame = function()
{
    if (this.manual) {
        this.frame = null;

        return;
    }

    clearTimeout(this.frame);
    this.frame = null;
};

/**
 * Schedule a callback
 *
 * @param {Function} callback
 * @param {Number} delay
 *
 * @return {Number|Object}
 */
BaseGame.prototype.scheduleTimeout = function(callback, delay)
{
    if (!this.manual) {
        return setTimeout(callback, delay);
    }

    var schedule = {
        id: ++this.scheduleId,
        callback: callback,
        due: this.clock + Math.max(0, delay)
    };

    this.schedules.push(schedule);

    return schedule.id;
};

/**
 * Cancel a scheduled callback
 *
 * @param {Number|Object} id
 */
BaseGame.prototype.clearScheduleTimeout = function(id)
{
    if (!this.manual) {
        clearTimeout(id);

        return;
    }

    for (var i = this.schedules.length - 1; i >= 0; i--) {
        if (this.schedules[i].id === id) {
            this.schedules.splice(i, 1);
        }
    }
};

/**
 * Run all due callbacks
 */
BaseGame.prototype.runSchedules = function()
{
    var executed = true,
        schedule,
        i;

    while (executed) {
        executed = false;

        this.schedules.sort(function (a, b) { return a.due > b.due ? 1 : (a.due < b.due ? -1 : 0); });

        for (i = 0; i < this.schedules.length; i++) {
            schedule = this.schedules[i];

            if (schedule.due > this.clock) {
                break;
            }

            this.schedules.splice(i, 1);
            schedule.callback();
            executed = true;
            break;
        }
    }
};

/**
 * Advance a manual game by ticks
 *
 * @param {Number} ticks
 *
 * @return {Boolean}
 */
BaseGame.prototype.advance = function(ticks)
{
    if (!this.manual) {
        return false;
    }

    ticks = Math.max(1, parseInt(ticks, 10) || 1);

    for (var i = 0; i < ticks; i++) {
        this.clock += this.fixedStep;
        this.runSchedules();

        if (this.frame) {
            this.rendered = this.clock;
            this.onFrame(this.fixedStep);
            this.fps.onFrame();
        }
    }

    return true;
};

/**
 * On frame
 *
 * @param {Number} step
 */
BaseGame.prototype.onFrame = function(step)
{
    this.tick++;
    this.update(step);
};

/**
 * Update game size
 */
BaseGame.prototype.setSize = function()
{
    this.size = this.getSize(this.getPresentAvatars().count());
};

/**
 * Get size by players
 *
 * @param {Number} players
 *
 * @return {Number}
 */
BaseGame.prototype.getSize = function(players)
{
    var square = this.perPlayerSize * this.perPlayerSize,
        size   = Math.sqrt(square + ((players - 1) * square / 5));

    return Math.round(size);
};

/**
 * Are all avatars ready?
 *
 * @return {Boolean}
 */
BaseGame.prototype.isReady = function()
{
    return this.getLoadingAvatars().isEmpty();
};

/**
 * Get still loading avatars
 *
 * @return {Collection}
 */
BaseGame.prototype.getLoadingAvatars = function()
{
    return this.avatars.filter(function () { return this.present && !this.ready; });
};

/**
 * Get alive avatars
 *
 * @return {Collection}
 */
BaseGame.prototype.getAliveAvatars = function()
{
    return this.avatars.filter(function () { return this.alive; });
};

/**
 * Get present avatars
 *
 * @return {Collection}
 */
BaseGame.prototype.getPresentAvatars = function()
{
    return this.avatars.filter(function () { return this.present; });
};

/**
 * Sort avatars
 *
 * @param {Object} avatars
 *
 * @return {Object}
 */
BaseGame.prototype.sortAvatars = function(avatars)
{
    avatars = typeof(avatars) !== 'undefined' ? avatars : this.avatars;

    avatars.sort(function (a, b) { return a.score > b.score ? -1 : (a.score < b.score ? 1 : 0); });

    return avatars;
};

/**
 * Set borderless
 *
 * @param {Boolean} borderless
 */
BaseGame.prototype.setBorderless = function(borderless)
{
    this.borderless = borderless ? true : false;
};

/**
 * Serialize
 *
 * @return {Object}
 */
BaseGame.prototype.serialize = function()
{
    return {
        name: this.name,
        players: this.avatars.map(function () { return this.serialize(); }).items,
        maxScore: this.maxScore
    };
};

/**
 * New round
 */
BaseGame.prototype.newRound = function(time)
{
    this.started = true;

    if (!this.inRound) {
        this.inRound = true;
        this.onRoundNew();
        this.scheduleTimeout(this.start, typeof(time) !== 'undefined' ? time : this.warmupTime);
    }
};

/**
 * Check end of round
 */
BaseGame.prototype.endRound = function()
{
    if (this.inRound) {
        this.inRound = false;
        this.onRoundEnd();
        this.scheduleTimeout(this.stop, this.warmdownTime);
    }
};

/**
 * FIN DU GAME
 */
BaseGame.prototype.end = function()
{
    if (this.started) {
        this.started = false;
        this.stop();
        this.emit('end', {game: this});

        return true;
    }

    return false;
};
