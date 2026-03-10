var fs = require('fs');
var path = require('path');
var glob = require('glob');

var recipe = require('./recipes/server.json');
var output = '';

recipe.files.forEach(function(pattern) {
    var files = glob.sync(pattern);
    files.sort();
    files.forEach(function(file) {
        output += fs.readFileSync(file, 'utf8') + '\n';
    });
});

var outDir = recipe.path;
if (!fs.existsSync(outDir)) {
    fs.mkdirSync(outDir, { recursive: true });
}

fs.writeFileSync(path.join(outDir, recipe.name), output);
console.log('Built ' + path.join(outDir, recipe.name) + ' (' + Math.round(output.length / 1024) + ' KB)');
