/**
 * Bot client
 */
function BotClient(name)
{
    EventEmitter.call(this);

    this.id = 'bot:' + (++BotClient.prototype.lastId);
    this.name = name || this.id;
    this.active = true;
    this.connected = true;
    this.players = new Collection([], 'id');
    this.events = [];
    this.pingLogger = {
        start: function () {},
        stop: function () {}
    };
}

BotClient.prototype = Object.create(EventEmitter.prototype);
BotClient.prototype.constructor = BotClient;

/**
 * Bot client counter
 *
 * @type {Number}
 */
BotClient.prototype.lastId = 0;

/**
 * Is this client playing?
 *
 * @return {Boolean}
 */
BotClient.prototype.isPlaying = function()
{
    return !this.players.isEmpty();
};

/**
 * Clear all players
 */
BotClient.prototype.clearPlayers = function()
{
    this.emit('players:clear', this);
    this.players.clear();
};

/**
 * Add a server event
 *
 * @param {String} name
 * @param {Object} data
 */
BotClient.prototype.addEvent = function(name, data)
{
    this.events.push([name, data]);
};

/**
 * Add several server events
 *
 * @param {Array} events
 */
BotClient.prototype.addEvents = function(events)
{
    for (var i = 0; i < events.length; i++) {
        this.events.push(events[i]);
    }
};

/**
 * Stop the client
 */
BotClient.prototype.stop = function()
{
    this.connected = false;
    this.emit('close', this);
};

/**
 * Serialize bot metadata like a normal client
 *
 * @return {Object}
 */
BotClient.prototype.serialize = function()
{
    return {
        id: this.id,
        active: this.active,
        bot: true,
        name: this.name
    };
};

