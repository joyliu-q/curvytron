var server = require('../bin/curvytron.js'),
    seed = process.argv[2] || 'demo-seed',
    session = server.rl.manager.createSession({
        seed: seed,
        grid_width: 32,
        grid_height: 32,
        max_score: 1,
        warmup_ms: 0,
        warmdown_ms: 0,
        print_delay_ms: 0
    }),
    actions = ['left', 'straight', 'right'],
    state,
    i;

if (!session) {
    throw new Error('Unable to create RL session.');
}

session.addBot({name: 'bot-a', color: '#66CCFF'});
session.addBot({name: 'bot-b', color: '#FF9966'});

state = session.startEpisode();

for (i = 0; i < 2000 && !state.done; i++) {
    var stepActions = {};

    for (var a = 0; a < session.actors.items.length; a++) {
        stepActions[session.actors.items[a].id] = actions[Math.floor(session.room.randomGenerator.next() * actions.length)];
    }

    state = session.step(stepActions);
}

console.log('Seed:', seed);
console.log('Ticks:', state.tick);
console.log('Winner:', state.round_winner_player_id);
console.log(state.occupancy.ascii);

session.close();
server.server.close();
