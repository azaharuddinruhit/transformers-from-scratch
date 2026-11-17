// Checks how two browsers' copies of the progress record merge for gist sync (progMerge in site.js):
// different items from both survive, the newer change to one item wins, deletions and resets travel.
//     node tools/test-sync.js
'use strict';
const fs = require('fs');
const src = fs.readFileSync(__dirname + '/../site/assets/js/site.js', 'utf8');
const a = src.indexOf('const PROG_KEY='), b = src.indexOf('const onProgress=');
if (a < 0 || b < a) throw new Error('progress functions not found in site.js');
const { progNorm, progMerge, progStamp, progSig } = new Function(src.slice(a, b) + ';return { progNorm, progMerge, progStamp, progSig };')();
const ok=(c,m)=>{console.log((c?'PASS ':'FAIL ')+m);if(!c)process.exitCode=1;};
const clone=x=>JSON.parse(JSON.stringify(x));
// an old record (no times) migrates with time 1
const old=progNorm({path:'quick',done:{tokens:true},quiz:{tokens:[0,1]},last:'part-b.html#tokens',drill:{'iv-core-1':'got'}});
ok(old.ts.done.tokens===1&&old.ts.quiz.tokens===1&&old.ts.last===1&&!('path' in old),'old record gets time 1 on every item, and its old path is dropped');
// two browsers change different things: both survive
let A=progNorm({}),B=progNorm({});let sa=clone(A),sb=clone(B);
A.done.tokens=true;A.quiz.tokens=[0];progStamp(A,sa,100);
B.done.attention=true;B.quiz.tokens=[2];B.drill['iv-rag-3']='again';progStamp(B,sb,200);
let M=progMerge(A,B);
ok(M.done.tokens&&M.done.attention&&M.drill['iv-rag-3']==='again','different sections from two browsers both kept');
ok(JSON.stringify(M.quiz.tokens)==='[0,2]','quiz answers of both copies are united');
ok(progSig(progMerge(B,A))===progSig(M),'merge is symmetric');
ok(progSig(progMerge(M,M))===progSig(M),'merging a copy with itself changes nothing');
// same item changed on both: newer wins, including un-marking done and deleting a drill mark
A=clone(M);B=clone(M);sa=clone(A);sb=clone(B);
A.done.tokens=false;progStamp(A,sa,300);
delete B.drill['iv-rag-3'];B.done.tokens=true;progStamp(B,sb,250);
M=progMerge(A,B);
ok(M.done.tokens===false,'newer change to the same section wins (un-marked at 300 beats 250)');
ok(!('iv-rag-3' in M.drill)&&M.ts.drill['iv-rag-3']===250,'a deleted drill mark stays deleted (its time is kept)');
// Continue place: newer wins
A=clone(M);sa=clone(A);A.last='part-c.html#flash';progStamp(A,sa,400);
M=progMerge(M,A);ok(M.last==='part-c.html#flash','the Continue place follows the newer copy');
// reset on one browser clears older items from the other and keeps later ones
const R=clone(M);R.done={};R.quiz={};R.drill={};R.last=null;R.resetAt=500;R.ts={last:0,done:{},quiz:{},drill:{}};
B=clone(M);sb=clone(B);B.done.flash=true;B.quiz.flash=[1];progStamp(B,sb,600);
M=progMerge(B,R);
ok(!M.done.tokens&&!M.done.attention&&!M.quiz.tokens&&M.last===null,'a reset clears everything changed before it, on both copies');
ok(M.done.flash===true&&JSON.stringify(M.quiz.flash)==='[1]','work done after the reset is kept');
ok(M.resetAt===500,'the reset time travels');
// round trip through JSON (what the gist stores)
ok(progSig(progNorm(JSON.parse(JSON.stringify(M))))===progSig(M),'a record survives the trip through the gist unchanged');
if(!process.exitCode)console.log('all merge checks passed');
