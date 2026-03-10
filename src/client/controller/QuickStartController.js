/**
 * Quick Start Controller
 *
 * Sends a quickstart request to the server, then redirects
 * to the room lobby with autostart enabled.
 *
 * @param {Object} $scope
 * @param {Object} $routeParams
 * @param {Object} $location
 * @param {SocketClient} client
 */
function QuickStartController($scope, $routeParams, $location, client)
{
    AbstractController.call(this, $scope);

    this.$location    = $location;
    this.client       = client;
    this.minPlayers   = parseInt($routeParams.n, 10) || 2;

    this.$scope.minPlayers = this.minPlayers;
    this.$scope.status     = 'Connecting...';

    // Binding
    this.quickStart = this.quickStart.bind(this);

    this.quickStart();
}

QuickStartController.prototype = Object.create(AbstractController.prototype);
QuickStartController.prototype.constructor = QuickStartController;

/**
 * Send quickstart request to server
 */
QuickStartController.prototype.quickStart = function()
{
    if (!this.client.connected) {
        this.$scope.status = 'Connecting...';
        return this.client.on('connected', this.quickStart);
    }

    var controller = this;

    this.$scope.status = 'Finding a game...';
    this.digestScope();

    this.client.addEvent(
        'room:quickstart',
        { players: this.minPlayers },
        function (result) {
            if (result.success) {
                controller.$location.path('/room/' + encodeURIComponent(result.name)).search('autostart', '1');
                controller.applyScope();
            } else {
                controller.$scope.status = result.error || 'Could not start game.';
                controller.digestScope();
            }
        }
    );
};
