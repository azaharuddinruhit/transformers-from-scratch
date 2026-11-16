'use strict';
/* Maths helpers and the illustrative next-token logits of the running example, used by more than one part. */
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
function randn(r){let u=0;while(u===0)u=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*r());}
function dot(a,b){let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s;}
function softmax(z,temp){temp=temp||1;let m=-Infinity;for(const v of z)if(v>m)m=v;const e=z.map(v=>Math.exp((v-m)/temp));let s=0;for(const v of e)s+=v;return e.map(v=>v/s);}
function erf(x){const s=x<0?-1:1;x=Math.abs(x);const t=1/(1+0.3275911*x);const y=1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-x*x);return s*y;}
function gelu(x){return 0.5*x*(1+erf(x/Math.SQRT2));}
function geluGrad(x){return 0.5*(1+erf(x/Math.SQRT2))+x*Math.exp(-x*x/2)/Math.sqrt(2*Math.PI);}
function sampleFrom(p,rng){const r=rng();let c=0,best=0;for(let i=0;i<p.length;i++){if(p[i]>p[best])best=i;c+=p[i];if(r<c)return i;}return best;}
/* ---------- hand-written illustrative logits ---------- */
const NEXT=[
  {text:'The cat sat on the',toks:['mat','sofa','floor','roof','dog','moon','banana','the'],logits:[3.2,2.6,2.4,1.5,0.4,-0.3,-1.5,-2.5]},
  {text:'I drank a cup of',toks:['coffee','tea','water','milk','soup','sand','the','mat'],logits:[3.4,3.0,2.2,1.6,0.6,-1.8,-2.5,-3.0]},
  {text:'Two plus two equals',toks:['four','4','five','three','two','fish','the','mat'],logits:[4.5,2.8,1.2,0.8,0.5,-1.5,-2.5,-3.0]}
];
