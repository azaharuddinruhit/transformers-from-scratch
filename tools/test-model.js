// Trains the toy model from Part C outside the browser, with the widget's defaults
// (120 sentences, seed 42, 600 steps, peak learning rate 0.005), and checks the validation loss.
//     node tools/test-model.js
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const js = path.join(__dirname, '..', 'site', 'assets', 'js');
const context = vm.createContext({});
for (const file of ['shared.js', 'toy-model.js']) {
  vm.runInContext(fs.readFileSync(path.join(js, file), 'utf8'), context, { filename: file });
}
const result = vm.runInContext(`(() => {
  const tr = createTrainer({ nTrain: 120, seed: 42 }), total = 600;
  const before = evalLoss(tr.model, tr.val);
  while (tr.step < total) trainStep(tr, lrAt(tr.step, 0, total, 0.005));
  return { before, after: evalLoss(tr.model, tr.val), sample: generate(tr.model, 'the cat ', 40, 0.8, 1) };
})()`, context);

console.log(`validation loss: ${result.before.toFixed(3)} untrained, ${result.after.toFixed(3)} after 600 steps`);
console.log(`sample: ${JSON.stringify(result.sample)}`);
if (!(result.after < 0.4)) {
  console.error('FAIL: the guide says training reaches a validation loss near 0.3 in about 600 steps');
  process.exit(1);
}
console.log('OK');
