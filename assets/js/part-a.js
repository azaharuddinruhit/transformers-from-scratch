'use strict';
/* Part A widgets: the next-token predictor and the attention-versus-recurrence cost comparison. */
/* ================= next-token predictor (What is an LLM?) ================= */
const s1={p:0,temp:1,counts:null,last:null};
function renderNext(){
  const P=NEXT[s1.p],pr=softmax(P.logits,s1.temp);if(!s1.counts)s1.counts=P.toks.map(()=>0);
  const total=s1.counts.reduce((a,b)=>a+b,0);
  $('#nx-chart').innerHTML=bars(P.toks.map((t,i)=>({label:esc(t),p:pr[i],cls:i===s1.last?'hit':'',right:fmt(pr[i])+(total?' · '+s1.counts[i]+'×':''),tip:'logit '+P.logits[i].toFixed(1)+' → probability '+pr[i].toFixed(4)})),1);
  $('#nx-out').innerHTML=esc(P.text)+' '+(s1.last==null?'<b>___</b>':'<mark>'+esc(P.toks[s1.last])+'</mark>')+(total?' <small>('+total+' sample'+(total>1?'s':'')+' drawn so far)</small>':'');
}
function initNext(){
  $('#nx-prompt').innerHTML=NEXT.map((p,i)=>'<option value="'+i+'">'+esc(p.text)+' …</option>').join('');
  $('#nx-prompt').addEventListener('change',e=>{s1.p=+e.target.value;s1.counts=null;s1.last=null;renderNext();});
  slider('nx-temp',v=>v.toFixed(1),v=>{s1.temp=v;renderNext();});
  $('#nx-sample').addEventListener('click',()=>{const pr=softmax(NEXT[s1.p].logits,s1.temp);s1.last=sampleFrom(pr,Math.random);if(!s1.counts)s1.counts=pr.map(()=>0);s1.counts[s1.last]++;renderNext();});
  $('#nx-reset').addEventListener('click',()=>{s1.counts=null;s1.last=null;renderNext();});
  reg(renderNext);
}
const cx={n:64,d:512};
function renderCX(){
  const n=cx.n,d=cx.d,sa=n*n*d,rn=n*d*d;
  $('#cx-bars').innerHTML=bars([{label:'self-attention',p:sa,right:sci(sa),tip:sym('T² · d_model = ')+n+'² × '+d+' = '+sa.toLocaleString('en-US')},{label:'recurrent',p:rn,right:sci(rn),tip:sym('T · d_model² = ')+n+' × '+d+'² = '+rn.toLocaleString('en-US')}],Math.max(sa,rn));
  $('#cx-table').innerHTML='<thead><tr><th>Layer type</th><th>Work per layer</th><th>Sequential steps</th><th>Longest path</th></tr></thead><tbody><tr><th scope="row">Self-attention</th><td>'+sym('T²·d_model')+' = '+sa.toLocaleString('en-US')+'</td><td>1</td><td>1</td></tr><tr><th scope="row">Recurrent</th><td>'+sym('T·d_model²')+' = '+rn.toLocaleString('en-US')+'</td><td>'+n.toLocaleString('en-US')+'</td><td>'+n.toLocaleString('en-US')+'</td></tr></tbody>';
  $('#cx-read').innerHTML=(n<d?'<span class="badge ok">T &lt; '+sym('d_model')+'</span> The sequence is shorter than the width, so self-attention does '+(rn/sa).toFixed(1)+'× less work per layer':n===d?'<span class="badge ok">T = '+sym('d_model')+'</span> The two layer types do the same amount of work per layer':'<span class="badge bad">T &gt; '+sym('d_model')+'</span> The sequence is longer than the width, so self-attention does '+(sa/rn).toFixed(1)+'× more work per layer')+', and the recurrent layer needs '+n.toLocaleString('en-US')+' steps one after another where self-attention needs 1.';
}
function initCX(){
  slider('cx-n',v=>Math.pow(2,v).toLocaleString('en-US'),v=>{cx.n=Math.pow(2,v);renderCX();});
  slider('cx-d',v=>Math.pow(2,v).toLocaleString('en-US'),v=>{cx.d=Math.pow(2,v);renderCX();});
  reg(renderCX);
}

PAGE_INITS.push(initNext,initCX);
