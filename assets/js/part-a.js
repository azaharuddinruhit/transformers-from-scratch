'use strict';
/* Part A widgets: the next-token predictor, a tiny trainable neural network, a name generator that learns by counting
   and the attention-versus-recurrence cost comparison. */
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
/* ================= a tiny neural network (Neural networks in 15 minutes) ================= */
/* 2 inputs -> h hidden units (tanh, or no activation) -> 1 sigmoid output, trained with full-batch gradient descent
   on the cross-entropy loss. Everything is computed here; nothing is pretrained. */
const NN_MAX=2000,NN_LR=0.5;
const NN={data:'ring',h:4,act:'tanh',n:80,step:0,timer:null,hist:[],net:null,train:null,val:null};
function nnData(kind,n,seed){
  const r=mulberry32(seed),pts=[],cl=v=>Math.max(-0.98,Math.min(0.98,v));
  for(let i=0;i<n;i++){
    let c=i%2,x,y;
    if(kind==='groups'){x=(c?0.42:-0.42)+0.3*randn(r);y=(c?0.32:-0.32)+0.3*randn(r);}
    else{const a=2*Math.PI*r(),rad=c?0.36*Math.sqrt(r()):0.74+0.08*randn(r);x=rad*Math.cos(a);y=rad*Math.sin(a);}
    if(r()<0.1)c=1-c;                       // one label in ten is wrong, as in real data
    pts.push([cl(x),cl(y),c]);
  }
  return pts;
}
function nnInit(h,seed){
  const r=mulberry32(seed),net={W1:[],b1:[],W2:[],b2:0};
  for(let j=0;j<h;j++){net.W1.push([0.9*randn(r),0.9*randn(r)]);net.b1.push(0);net.W2.push(0.9*randn(r));}
  return net;
}
function nnOut(net,x,y,act){
  let z=net.b2;
  for(let j=0;j<net.W2.length;j++){const a=net.W1[j][0]*x+net.W1[j][1]*y+net.b1[j];z+=net.W2[j]*(act==='tanh'?Math.tanh(a):a);}
  return 1/(1+Math.exp(-z));
}
function nnLoss(net,pts,act){
  let L=0,ok=0;
  pts.forEach(p=>{const q=nnOut(net,p[0],p[1],act);L-=Math.log(Math.max(1e-9,p[2]?q:1-q));if((q>0.5)===(p[2]===1))ok++;});
  return[L/pts.length,ok/pts.length];
}
/* one step of gradient descent: forward pass, backward pass (the chain rule), update */
function nnStep(net,pts,act,lr){
  const h=net.W2.length,g1=net.W1.map(()=>[0,0]),gb1=net.b1.map(()=>0),g2=net.W2.map(()=>0);let gb2=0;
  pts.forEach(p=>{
    const hid=[];let z=net.b2;
    for(let j=0;j<h;j++){const a=net.W1[j][0]*p[0]+net.W1[j][1]*p[1]+net.b1[j];hid[j]=act==='tanh'?Math.tanh(a):a;z+=net.W2[j]*hid[j];}
    const d=1/(1+Math.exp(-z))-p[2];          // slope of the loss with respect to z
    gb2+=d;
    for(let j=0;j<h;j++){g2[j]+=d*hid[j];const dh=d*net.W2[j]*(act==='tanh'?1-hid[j]*hid[j]:1);g1[j][0]+=dh*p[0];g1[j][1]+=dh*p[1];gb1[j]+=dh;}
  });
  const s=lr/pts.length;
  for(let j=0;j<h;j++){net.W2[j]-=s*g2[j];net.W1[j][0]-=s*g1[j][0];net.W1[j][1]-=s*g1[j][1];net.b1[j]-=s*gb1[j];}
  net.b2-=s*gb2;
}
function nnReset(){
  const s=NN;if(s.timer){clearInterval(s.timer);s.timer=null;}
  s.train=nnData(s.data,s.n,s.data==='ring'?11:23);s.val=nnData(s.data,60,s.data==='ring'?97:71);
  s.net=nnInit(s.h,5);s.step=0;s.hist=[[0,nnLoss(s.net,s.train,s.act)[0],nnLoss(s.net,s.val,s.act)[0]]];
}
function nnAdvance(k){
  const s=NN;
  for(let i=0;i<k&&s.step<NN_MAX;i++){nnStep(s.net,s.train,s.act,NN_LR);s.step++;}
  s.hist.push([s.step,nnLoss(s.net,s.train,s.act)[0],nnLoss(s.net,s.val,s.act)[0]]);
}
function renderNN(){
  const s=NN,G=24,c=240/G;let g='';
  for(let i=0;i<G;i++)for(let j=0;j<G;j++){const q=nnOut(s.net,-1+(i+0.5)*2/G,1-(j+0.5)*2/G,s.act);g+='<rect x="'+(i*c)+'" y="'+(j*c)+'" width="'+(c+0.5)+'" height="'+(c+0.5)+'" style="fill:color-mix(in srgb,var(--accent) '+Math.round(62*q)+'%,var(--panel))"/>';}
  const P=(p,k)=>'<circle class="np np'+p[2]+' '+k+'" cx="'+((p[0]+1)*120).toFixed(1)+'" cy="'+((1-p[1])*120).toFixed(1)+'" r="'+(k==='t'?4.2:3.6)+'"/>';
  $('#nn-svg').innerHTML=g+s.val.map(p=>P(p,'v')).join('')+s.train.map(p=>P(p,'t')).join('');
  const tr=nnLoss(s.net,s.train,s.act),va=nnLoss(s.net,s.val,s.act);
  lineChart($('#nn-chart'),[{pts:s.hist.map(h=>[h[0],h[2]]),cls:'b'},{pts:s.hist.map(h=>[h[0],h[1]]),cls:'a'}],{x0:0,x1:NN_MAX,y0:0,y1:1.2,h:200,xt:[0,500,1000,1500,2000],yt:[0,0.4,0.8,1.2],xl:'training step',yl:'loss'});
  $('#nn-stats').innerHTML=[['Step',s.step+' / '+NN_MAX],['Parameters',4*s.h+1],['Training loss',fmt(tr[0],3)],['Validation loss',fmt(va[0],3)],['Training accuracy',Math.round(tr[1]*100)+'%'],['Validation accuracy',Math.round(va[1]*100)+'%']].map(r=>'<div><b>'+r[1]+'</b>'+r[0]+'</div>').join('');
  $('#nn-play').textContent=s.timer?'❚❚ Pause':s.step>=NN_MAX?'↺ Train again':'▶ Train';
  [['nn-data',s.data],['nn-h',String(s.h)],['nn-act',s.act],['nn-n',String(s.n)]].forEach(([id,v])=>$$('#'+id+' button').forEach(b=>{const on=b.dataset.v===v;b.classList.toggle('on',on);b.setAttribute('aria-pressed',on);}));
}
function initNN(){
  const chips=(id,opts,set)=>{const el=$('#'+id);el.innerHTML=opts.map(o=>'<button class="chip" data-v="'+o[0]+'" aria-pressed="false">'+o[1]+'</button>').join('');
    el.addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;set(b.dataset.v);nnReset();renderNN();const n=$('#'+id+' button[data-v="'+b.dataset.v+'"]');if(n)n.focus();});};
  chips('nn-data',[['groups','Two groups'],['ring','Ring']],v=>{NN.data=v;});
  chips('nn-h',[['1','1'],['2','2'],['4','4'],['16','16']],v=>{NN.h=+v;});
  chips('nn-act',[['tanh','tanh'],['none','None']],v=>{NN.act=v;});
  chips('nn-n',[['12','12'],['80','80']],v=>{NN.n=+v;});
  const stop=()=>{if(NN.timer){clearInterval(NN.timer);NN.timer=null;}};
  $('#nn-play').addEventListener('click',()=>{
    if(NN.timer){stop();renderNN();return;}
    if(NN.step>=NN_MAX)nnReset();
    NN.timer=setInterval(()=>{nnAdvance(20);if(NN.step>=NN_MAX)stop();renderNN();},60);renderNN();});
  $('#nn-step').addEventListener('click',()=>{stop();if(NN.step<NN_MAX)nnAdvance(1);renderNN();});
  $('#nn-reset').addEventListener('click',()=>{nnReset();renderNN();});
  nnReset();reg(renderNN);
}
/* ================= a generator that learns by counting (Generative AI: the big picture) ================= */
/* A character n-gram table built from GEN_NAMES when the page loads; nothing is pretrained. "." marks start and end. */
const GEN_NAMES=('aaron abdul adam adrian aisha akira alba alex alice alina amara amir amy ana andrea andrei anika anna anton arjun ava ayesha '+
  'beatriz ben bianca boris bruno carla carlos carmen caroline chen chloe clara daniel david diana diego dmitri elena eli elias elif '+
  'elise ella emil emily emma erik esther ethan eva farah felix fiona frida gabriel grace hana hannah hassan hector helen hugo ibrahim '+
  'ida imran ines ingrid irene isaac isabel ivan jack jacob james jana jasmine javier jonas jose julia julian kai karim karin karl '+
  'kate kenji kira lara laura layla leo leon lena liam lina linus lucas lucia luis luna maja malik marco maria marina mark marta '+
  'martin maya mia mila mina miriam mohammed nadia naomi nathan nina noah nora olga oliver omar oscar pablo paula pedro peter '+
  'priya rafael rahul rania ravi rina rosa ruby rupert sara sami samuel sana sofia sonia stefan sven tara theo thomas tomas '+
  'valentina vera victor viktor william yara yasmin yusuf zara zoe').split(' ');
const GEN_SET=new Set(GEN_NAMES),GEN_MAX=14;
const GN={k:2,word:'',done:false,list:[],tables:{}};
function genTable(k){
  if(GN.tables[k])return GN.tables[k];
  const T={};
  GEN_NAMES.forEach(w=>{const s='.'.repeat(k)+w+'.';for(let i=k;i<s.length;i++){const c=s.slice(i-k,i),row=T[c]||(T[c]={});row[s[i]]=(row[s[i]]||0)+1;}});
  return GN.tables[k]=T;
}
const genCtx=()=>('.'.repeat(GN.k)+GN.word).slice(-GN.k);
/* one letter: draw from the counts for the current context; the end marker (or the length cap) finishes the name */
function genStep(){
  if(GN.done){GN.word='';GN.done=false;}
  const row=genTable(GN.k)[genCtx()],keys=Object.keys(row),tot=keys.reduce((a,c)=>a+row[c],0);
  let u=Math.random()*tot,ch=keys[keys.length-1];
  for(const c of keys){u-=row[c];if(u<0){ch=c;break;}}
  if(ch==='.'||GN.word.length>=GEN_MAX){GN.done=true;GN.list.unshift(GN.word);}else GN.word+=ch;
}
function genName(){do genStep();while(!GN.done);}
function renderNames(){
  const k=GN.k,ctx=genCtx(),row=genTable(k)[ctx],tot=Object.values(row).reduce((a,b)=>a+b,0);
  const letters=ctx.replace(/\./g,''),after=letters?'after “'+letters+'”'+(ctx[0]==='.'?' at the start of a name':''):'for the first letter of a name';
  const w=GN.word,cut=Math.max(0,w.length-k);
  if(GN.done)$('#gn-word').innerHTML='<code>'+esc(w)+'</code> '+(GEN_SET.has(w)?'<span class="badge bad">a copy: this name is in the training list</span>':'<span class="badge ok">new: not in the training list</span>');
  else $('#gn-word').innerHTML=w?'<code>'+esc(w.slice(0,cut))+'<mark>'+esc(w.slice(cut))+'</mark>▢</code> <small>The marked letters are the context for the next draw.</small>':'Press “Next letter” to draw the first letter, or “Generate a name”.';
  $('#gn-lab').textContent=(GN.done?'Probabilities for the first letter of the next name':'Probabilities for the next letter, '+after)+' (counted from '+tot+' cases in the training names)';
  const rows=Object.keys(row).map(c=>({c,n:row[c]})).sort((a,b)=>b.n-a.n),top=rows.slice(0,8),rest=rows.slice(8);
  const bl=top.map(r=>({label:r.c==='.'?'end':esc(r.c),p:r.n/tot,right:fmt(r.n/tot,2),tip:'In the training names, '+(r.c==='.'?'the name ends':'“'+r.c+'” comes next')+' '+r.n+' of '+tot+' times '+esc(after)}));
  if(rest.length){const n=rest.reduce((a,r)=>a+r.n,0);bl.push({label:'other',p:n/tot,right:fmt(n/tot,2),cls:'cut',tip:rest.length+' more letters: '+rest.map(r=>r.c==='.'?'end':r.c).join(', ')});}
  $('#gn-bars').innerHTML=bars(bl,1);
  $('#gn-list').innerHTML=GN.list.length?GN.list.slice(0,30).map(n=>'<span class="chip tok">'+esc(n)+'<small>'+(GEN_SET.has(n)?'copy':'new')+'</small></span>').join(''):'<small>None yet.</small>';
  const N=GN.list.length,cp=GN.list.filter(n=>GEN_SET.has(n)).length;
  $('#gn-stats').innerHTML=[['Names generated',N],['New names',N?Math.round(100*(N-cp)/N)+'%':'–'],['Copies',N?Math.round(100*cp/N)+'%':'–'],['Contexts in the table',Object.keys(genTable(k)).length]].map(r=>'<div><b>'+r[1]+'</b><span>'+r[0]+'</span></div>').join('');
  $$('#gn-k button').forEach(b=>{const on=+b.dataset.v===k;b.classList.toggle('on',on);b.setAttribute('aria-pressed',on);});
}
function initNames(){
  const el=$('#gn-k');
  el.innerHTML=[1,2,3,4].map(v=>'<button class="chip" data-v="'+v+'" aria-pressed="false">'+v+'</button>').join('');
  el.addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;GN.k=+b.dataset.v;GN.word='';GN.done=false;GN.list=[];renderNames();});
  $('#gn-one').addEventListener('click',()=>{if(GN.done)GN.word='';GN.done=false;genName();renderNames();});
  $('#gn-step').addEventListener('click',()=>{genStep();renderNames();});
  $('#gn-many').addEventListener('click',()=>{for(let i=0;i<20;i++){GN.word='';GN.done=false;genName();}renderNames();});
  $('#gn-clear').addEventListener('click',()=>{GN.word='';GN.done=false;GN.list=[];renderNames();});
  reg(renderNames);
}

PAGE_INITS.push(initNext,initNN,initNames,initCX);
