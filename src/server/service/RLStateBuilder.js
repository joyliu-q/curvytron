/**
 * RL state builder
 */
function RLStateBuilder() {}

/**
 * Empty cell value
 *
 * @type {Number}
 */
RLStateBuilder.prototype.empty = 0;

/**
 * Solid occupancy value
 *
 * @type {Number}
 */
RLStateBuilder.prototype.solid = 1;

/**
 * Bonus occupancy base value
 *
 * @type {Number}
 */
RLStateBuilder.prototype.bonus = 2;

/**
 * Player occupancy base value
 *
 * @type {Number}
 */
RLStateBuilder.prototype.playerBase = 100;

/**
 * Stable supported bonus type order
 *
 * @type {Array}
 */
RLStateBuilder.prototype.bonusTypes = [
    'BonusSelfSmall',
    'BonusSelfSlow',
    'BonusSelfFast',
    'BonusSelfMaster',
    'BonusEnemySlow',
    'BonusEnemyFast',
    'BonusEnemyBig',
    'BonusEnemyInverse',
    'BonusEnemyStraightAngle',
    'BonusGameBorderless',
    'BonusAllColor',
    'BonusGameClear'
];

/**
 * Stable ASCII markers per bonus type
 *
 * @type {Object}
 */
RLStateBuilder.prototype.bonusMarkers = {
    BonusSelfSmall: 's',
    BonusSelfSlow: 'l',
    BonusSelfFast: 'f',
    BonusSelfMaster: 'm',
    BonusEnemySlow: 'w',
    BonusEnemyFast: 't',
    BonusEnemyBig: 'b',
    BonusEnemyInverse: 'i',
    BonusEnemyStraightAngle: 'a',
    BonusGameBorderless: 'o',
    BonusAllColor: 'c',
    BonusGameClear: 'x'
};

/**
 * Build a state snapshot for the given session
 *
 * @param {RLSession} session
 * @param {Object} options
 *
 * @return {Object}
 */
RLStateBuilder.prototype.build = function(session, options)
{
    options = options || {};

    var gridWidth = this.getGridSize(options.gridWidth, session.grid.width),
        gridHeight = this.getGridSize(options.gridHeight, session.grid.height),
        game = session.room.game,
        grid = this.createGrid(gridWidth, gridHeight),
        players = [],
        bonuses = [],
        seen = {},
        actorIndex = 0,
        legend,
        i;

    if (!game) {
        for (i = 0; i < session.room.players.items.length; i++) {
            players.push(this.serializePlayer(session.room.players.items[i], null, session, actorIndex));
            actorIndex++;
        }

        return {
            session_id: session.id,
            mode: session.mode,
            seed: session.seed,
            tick: session.lastTick,
            in_round: false,
            done: session.done,
            round_winner_player_id: session.lastResult.round_winner_player_id,
            alive_player_ids: session.lastResult.alive_player_ids || [],
            board: {
                size: null,
                borderless: false
            },
            players: players,
            bonuses: [],
            occupancy: {
                width: gridWidth,
                height: gridHeight,
                cells: grid.cells,
                ascii: this.toAscii(grid.chars),
                legend: this.buildLegend(players)
            }
        };
    }

    if (!game.borderless) {
        this.markBorder(grid);
    }

    for (i = game.world.islands.items.length - 1; i >= 0; i--) {
        this.markIslandBodies(game, game.world.islands.items[i], seen, grid);
    }

    for (i = 0; i < game.bonusManager.bonuses.items.length; i++) {
        bonuses.push(this.serializeBonus(game.bonusManager.bonuses.items[i]));
        this.markDisc(
            grid,
            game,
            game.bonusManager.bonuses.items[i].x,
            game.bonusManager.bonuses.items[i].y,
            BaseBonus.prototype.radius,
            this.getBonusValue(game.bonusManager.bonuses.items[i].constructor.name),
            this.getBonusMarker(game.bonusManager.bonuses.items[i].constructor.name)
        );
    }

    for (i = 0; i < session.room.players.items.length; i++) {
        players.push(this.serializePlayer(session.room.players.items[i], game, session, actorIndex));

        if (session.room.players.items[i].avatar && session.room.players.items[i].avatar.alive) {
            this.markDisc(
                grid,
                game,
                session.room.players.items[i].avatar.x,
                session.room.players.items[i].avatar.y,
                session.room.players.items[i].avatar.radius * 2,
                this.getPlayerValue(actorIndex),
                this.getPlayerMarker(actorIndex)
            );
        }

        actorIndex++;
    }

    legend = this.buildLegend(players);

    return {
        session_id: session.id,
        mode: session.mode,
        seed: session.seed,
        tick: game.tick,
        in_round: game.inRound,
        done: session.done,
        round_winner_player_id: session.lastResult.round_winner_player_id,
        alive_player_ids: game.getAliveAvatars().map(function () { return this.id; }).items,
        board: {
            size: game.size,
            borderless: game.borderless
        },
        players: players,
        bonuses: bonuses,
        occupancy: {
            width: gridWidth,
            height: gridHeight,
            cells: grid.cells,
            ascii: this.toAscii(grid.chars),
            legend: legend
        }
    };
};

/**
 * Clamp grid size
 *
 * @param {Number} requested
 * @param {Number} fallback
 *
 * @return {Number}
 */
RLStateBuilder.prototype.getGridSize = function(requested, fallback)
{
    requested = parseInt(requested, 10);

    if (!requested || requested < 8) {
        requested = fallback;
    }

    return Math.min(128, requested);
};

/**
 * Create an empty occupancy grid
 *
 * @param {Number} width
 * @param {Number} height
 *
 * @return {Object}
 */
RLStateBuilder.prototype.createGrid = function(width, height)
{
    var cells = [],
        chars = [];

    for (var y = 0; y < height; y++) {
        cells[y] = [];
        chars[y] = [];

        for (var x = 0; x < width; x++) {
            cells[y][x] = this.empty;
            chars[y][x] = '.';
        }
    }

    return {
        cells: cells,
        chars: chars
    };
};

/**
 * Mark world bodies from an island
 *
 * @param {Game} game
 * @param {Island} island
 * @param {Object} seen
 * @param {Object} grid
 */
RLStateBuilder.prototype.markIslandBodies = function(game, island, seen, grid)
{
    for (var i = 0; i < island.bodies.items.length; i++) {
        var body = island.bodies.items[i];

        if (!seen[body.id]) {
            seen[body.id] = true;
            this.markDisc(grid, game, body.x, body.y, body.radius, this.solid, '#');
        }
    }
};

/**
 * Mark the board border
 *
 * @param {Object} grid
 */
RLStateBuilder.prototype.markBorder = function(grid)
{
    var x, y,
        maxX = grid.cells[0].length - 1,
        maxY = grid.cells.length - 1;

    for (x = 0; x <= maxX; x++) {
        grid.cells[0][x] = this.solid;
        grid.chars[0][x] = '#';
        grid.cells[maxY][x] = this.solid;
        grid.chars[maxY][x] = '#';
    }

    for (y = 0; y <= maxY; y++) {
        grid.cells[y][0] = this.solid;
        grid.chars[y][0] = '#';
        grid.cells[y][maxX] = this.solid;
        grid.chars[y][maxX] = '#';
    }
};

/**
 * Mark a disc in the downsampled grid
 *
 * @param {Object} grid
 * @param {Game} game
 * @param {Number} x
 * @param {Number} y
 * @param {Number} radius
 * @param {Number} value
 * @param {String} marker
 */
RLStateBuilder.prototype.markDisc = function(grid, game, x, y, radius, value, marker)
{
    var width = grid.cells[0].length,
        height = grid.cells.length,
        cellW = game.size / width,
        cellH = game.size / height,
        effectiveRadius = Math.max(radius, Math.max(cellW, cellH) * 0.75),
        minX = Math.max(0, Math.floor((x - effectiveRadius) / game.size * width)),
        maxX = Math.min(width - 1, Math.ceil((x + effectiveRadius) / game.size * width)),
        minY = Math.max(0, Math.floor((y - effectiveRadius) / game.size * height)),
        maxY = Math.min(height - 1, Math.ceil((y + effectiveRadius) / game.size * height)),
        marked = false;

    for (var gridY = minY; gridY <= maxY; gridY++) {
        for (var gridX = minX; gridX <= maxX; gridX++) {
            var worldX = (gridX + 0.5) * cellW,
                worldY = (gridY + 0.5) * cellH,
                distance = Math.sqrt(Math.pow(worldX - x, 2) + Math.pow(worldY - y, 2));

            if (distance <= effectiveRadius) {
                grid.cells[gridY][gridX] = value;
                grid.chars[gridY][gridX] = marker;
                marked = true;
            }
        }
    }

    if (!marked) {
        var cx = Math.max(0, Math.min(width - 1, Math.floor(x / game.size * width))),
            cy = Math.max(0, Math.min(height - 1, Math.floor(y / game.size * height)));

        grid.cells[cy][cx] = value;
        grid.chars[cy][cx] = marker;
    }
};

/**
 * Get a stable occupancy value for a bonus type
 *
 * @param {String} type
 *
 * @return {Number}
 */
RLStateBuilder.prototype.getBonusValue = function(type)
{
    var index = this.bonusTypes.indexOf(type);

    return index === -1 ? this.bonus + this.bonusTypes.length : this.bonus + index;
};

/**
 * Get a stable ASCII marker for a bonus type
 *
 * @param {String} type
 *
 * @return {String}
 */
RLStateBuilder.prototype.getBonusMarker = function(type)
{
    return this.bonusMarkers[type] || '?';
};

/**
 * Get a stable occupancy value for a player slot
 *
 * @param {Number} index
 *
 * @return {Number}
 */
RLStateBuilder.prototype.getPlayerValue = function(index)
{
    return this.playerBase + index;
};

/**
 * Serialize a player
 *
 * @param {Player} player
 * @param {Game} game
 * @param {RLSession} session
 * @param {Number} actorIndex
 *
 * @return {Object}
 */
RLStateBuilder.prototype.serializePlayer = function(player, game, session, actorIndex)
{
    var avatar = player.avatar,
        botActor = session.actors.getById(player.id);

    return {
        player_id: player.id,
        client_id: player.client.id,
        controller: botActor ? 'bot' : 'human',
        ready: player.ready,
        name: player.name,
        color: player.color,
        alive: avatar ? avatar.alive : false,
        present: avatar ? avatar.present : false,
        x: avatar ? avatar.x : null,
        y: avatar ? avatar.y : null,
        angle: avatar ? avatar.angle : null,
        velocity: avatar ? avatar.velocity : null,
        angular_velocity: avatar ? avatar.angularVelocity : null,
        radius: avatar ? avatar.radius : null,
        printing: avatar ? avatar.printing : null,
        inverse: avatar ? avatar.inverse : null,
        invincible: avatar ? avatar.invincible : null,
        score: avatar ? avatar.score : 0,
        round_score: avatar ? avatar.roundScore : 0,
        active_bonuses: avatar ? this.serializeActiveBonuses(avatar) : [],
        marker: this.getPlayerMarker(actorIndex),
        occupancy_value: this.getPlayerValue(actorIndex)
    };
};

/**
 * Serialize active bonuses on an avatar
 *
 * @param {Avatar} avatar
 *
 * @return {Array}
 */
RLStateBuilder.prototype.serializeActiveBonuses = function(avatar)
{
    var now = Date.now(),
        bonuses = avatar.bonusStack.bonuses.items,
        result = [],
        bonus, type, elapsed, remaining;

    for (var i = 0; i < bonuses.length; i++) {
        bonus = bonuses[i];
        type = bonus.constructor.name;
        elapsed = bonus.appliedAt ? now - bonus.appliedAt : 0;
        remaining = bonus.duration ? Math.max(0, bonus.duration - elapsed) : null;

        result.push({
            type: type,
            affect: bonus.affect,
            duration: bonus.duration,
            remaining_ms: remaining,
            marker: this.getBonusMarker(type)
        });
    }

    return result;
};

/**
 * Serialize a bonus
 *
 * @param {Bonus} bonus
 *
 * @return {Object}
 */
RLStateBuilder.prototype.serializeBonus = function(bonus)
{
    var type = bonus.constructor.name;

    return {
        id: bonus.id,
        type: type,
        x: bonus.x,
        y: bonus.y,
        marker: this.getBonusMarker(type),
        occupancy_value: this.getBonusValue(type)
    };
};

/**
 * Build a legend for ASCII and occupancy values
 *
 * @param {Array} players
 *
 * @return {Object}
 */
RLStateBuilder.prototype.buildLegend = function(players)
{
    var bonuses = {};

    for (var i = 0; i < this.bonusTypes.length; i++) {
        bonuses[this.bonusTypes[i]] = {
            marker: this.getBonusMarker(this.bonusTypes[i]),
            value: this.getBonusValue(this.bonusTypes[i])
        };
    }

    bonuses.unknown = {
        marker: '?',
        value: this.bonus + this.bonusTypes.length
    };

    return {
        empty: {
            marker: '.',
            value: this.empty
        },
        solid: {
            marker: '#',
            value: this.solid
        },
        bonuses: bonuses,
        players: players.map(function (player) {
            return {
                player_id: player.player_id,
                marker: player.marker,
                value: player.occupancy_value
            };
        })
    };
};

/**
 * Get a stable marker for a player slot
 *
 * @param {Number} index
 *
 * @return {String}
 */
RLStateBuilder.prototype.getPlayerMarker = function(index)
{
    var code = 65 + (index % 26);

    return String.fromCharCode(code);
};

/**
 * Convert char grid to ASCII text
 *
 * @param {Array} chars
 *
 * @return {String}
 */
RLStateBuilder.prototype.toAscii = function(chars)
{
    return chars.map(function (row) { return row.join(''); }).join('\n');
};
