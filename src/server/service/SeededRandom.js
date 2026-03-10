/**
 * Deterministic pseudo-random generator
 *
 * @param {String|Number} seed
 */
function SeededRandom(seed)
{
    this.seed = this.normalize(seed);
    this.state = this.seed;
}

/**
 * Normalize seed input to an unsigned 32-bit int
 *
 * @param {String|Number} seed
 *
 * @return {Number}
 */
SeededRandom.prototype.normalize = function(seed)
{
    if (typeof(seed) === 'number' && isFinite(seed)) {
        return seed >>> 0;
    }

    seed = typeof(seed) === 'undefined' || seed === null ? '0' : seed.toString();

    var hash = 2166136261;

    for (var i = 0; i < seed.length; i++) {
        hash ^= seed.charCodeAt(i);
        hash += (hash << 1) + (hash << 4) + (hash << 7) + (hash << 8) + (hash << 24);
    }

    return hash >>> 0;
};

/**
 * Next random float in [0, 1)
 *
 * @return {Float}
 */
SeededRandom.prototype.next = function()
{
    this.state += 0x6D2B79F5;

    var value = this.state;

    value = Math.imul(value ^ value >>> 15, value | 1);
    value ^= value + Math.imul(value ^ value >>> 7, value | 61);

    return ((value ^ value >>> 14) >>> 0) / 4294967296;
};

