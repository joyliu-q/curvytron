/**
 * Server-side controller for a bot player
 *
 * @param {Player} player
 */
function BotController(player)
{
    this.player = player;
    this.action = 'straight';
    this.move = 0;
    this.holdTimeout = null;
}

/**
 * Normalize an action value
 *
 * @param {String} action
 *
 * @return {String}
 */
BotController.prototype.normalizeAction = function(action)
{
    action = typeof(action) === 'string' ? action.toLowerCase() : 'straight';

    if (action === 'left' || action === 'right' || action === 'straight') {
        return action;
    }

    return 'straight';
};

/**
 * Convert an action to the game's move value
 *
 * @param {String} action
 *
 * @return {Number}
 */
BotController.prototype.getMove = function(action)
{
    switch (this.normalizeAction(action)) {
        case 'left':
            return -1;
        case 'right':
            return 1;
        default:
            return 0;
    }
};

/**
 * Get the current game instance
 *
 * @return {Game|null}
 */
BotController.prototype.getGame = function()
{
    return this.player && this.player.avatar ? this.player.avatar.game : null;
};

/**
 * Apply the currently selected action to the avatar
 */
BotController.prototype.sync = function()
{
    if (this.player && this.player.avatar) {
        this.player.avatar.updateAngularVelocity(this.move);
    }
};

/**
 * Set the current action
 *
 * @param {String} action
 *
 * @return {String}
 */
BotController.prototype.setAction = function(action)
{
    this.action = this.normalizeAction(action);
    this.move = this.getMove(this.action);
    this.sync();

    return this.action;
};

/**
 * Hold an action for a number of internal ticks
 *
 * @param {String} action
 * @param {Number} ticks
 *
 * @return {String}
 */
BotController.prototype.setAndHold = function(action, ticks)
{
    var game = this.getGame();

    this.setAction(action);

    if (!game) {
        return this.action;
    }

    if (this.holdTimeout) {
        game.clearScheduleTimeout(this.holdTimeout);
        this.holdTimeout = null;
    }

    ticks = Math.max(1, parseInt(ticks, 10) || 1);
    this.holdTimeout = game.scheduleTimeout(function () {
        this.holdTimeout = null;
        this.setAction('straight');
    }.bind(this), ticks * game.fixedStep);

    return this.action;
};

