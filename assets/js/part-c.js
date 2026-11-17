'use strict';
/* Part C widgets: training, post-training, generation, inference and GPUs. */
/* ================= training ================= */
let trainer=null;
const s8={running:false,start:0,total:600,lr:0.005,steps:600,temp:0.8,ema:[],val:[],last:null,cur:0,diverged:false,shown:-1};
function newTrainer(){trainer=createTrainer({nTrain:+$('#tr-size').value,seed:42});Object.assign(s8,{running:false,start:0,total:s8.steps,ema:[],val:[[0,evalLoss(trainer.model,trainer.val)]],last:null,cur:0,diverged:false,shown:-1});}
function probe(id,prompt){
  const p=softmax(nextLogits(trainer.model,encode(prompt))),ord=p.map((_,i)=>i).sort((a,b)=>p[b]-p[a]).slice(0,5);
  $('#'+id).innerHTML=bars(ord.map(i=>({label:'“'+showSp(CHARS[i])+'”',p:p[i]})),1);
}
function renderTrain(){
  const m=trainer.model,T=m.cfg.T,lnV=Math.log(CHARS.length),done=trainer.step>=s8.total,v=s8.val[s8.val.length-1][1];
  $('#tr-go').textContent=s8.running?'⏸ Pause':trainer.step===0?'▶ Train':done?'▶ Train '+s8.steps+' more steps':'▶ Resume';
  $('#tr-stats').innerHTML=[['Step',trainer.step+' / '+s8.total],['Epoch',(trainer.step*TOY_CFG.B*T/trainer.train.length).toFixed(1)],['Learning rate',trainer.step?s8.cur.toExponential(1):'—'],['Train loss',s8.last==null?'—':fmt(s8.last,3)],['Validation loss',fmt(v,3)],['Status',s8.diverged?'diverged':s8.running?'training…':trainer.step===0?'untrained':done?'finished':'paused']].map(r=>'<div><span>'+r[0]+'</span><b>'+r[1]+'</b></div>').join('');
  const xmax=Math.max(100,s8.total),stride=Math.max(1,Math.ceil(s8.ema.length/500)),pts=s8.ema.filter((_,i)=>i%stride===0||i===s8.ema.length-1);
  lineChart($('#tr-chart'),[{pts:[[0,lnV],[xmax,lnV]],cls:'c'},{pts:pts,cls:'a'},{pts:s8.val,cls:'b'}],{x0:0,x1:xmax,y0:0,y1:3.4,h:260,xt:[0,xmax/4,xmax/2,3*xmax/4,xmax].map(Math.round),yt:[0,1,2,3],xl:'training step',yl:'loss (nats per character)'});
  if(s8.shown!==trainer.step&&(!s8.running||trainer.step-s8.shown>=25||s8.shown<0)){
    s8.shown=trainer.step;probe('tr-probe1','the dog sa');probe('tr-probe2','a bird ran to ');
    $('#tr-sample').textContent=generate(m,'the ',90,s8.temp,7);
    if(typeof renderGenStatus==='function')renderGenStatus();
  }
}
function trainLoop(){
  if(!s8.running)return;const t0=performance.now();
  while(performance.now()-t0<16&&trainer.step<s8.total){
    s8.cur=lrAt(trainer.step,s8.start,s8.total,s8.lr);
    const loss=trainStep(trainer,s8.cur);
    if(!isFinite(loss)){s8.running=false;s8.diverged=true;break;}
    const prev=s8.ema.length?s8.ema[s8.ema.length-1][1]:loss;s8.last=0.9*prev+0.1*loss;s8.ema.push([trainer.step,s8.last]);
    if(trainer.step%25===0||trainer.step===s8.total)s8.val.push([trainer.step,evalLoss(trainer.model,trainer.val)]);
  }
  if(trainer.step>=s8.total)s8.running=false;
  renderTrain();
  if(s8.running)requestAnimationFrame(trainLoop);
}
function toggleTrain(){
  if(s8.running){s8.running=false;renderTrain();return;}
  if(s8.diverged)newTrainer();
  if(trainer.step>=s8.total){s8.start=trainer.step;s8.total=trainer.step+s8.steps;}
  s8.running=true;requestAnimationFrame(trainLoop);
}
function initTrain(){
  slider('tr-lr',v=>Math.pow(10,v).toPrecision(2),v=>{s8.lr=Math.pow(10,v);});s8.lr=Math.pow(10,+$('#tr-lr').value);
  slider('tr-steps',v=>v,v=>{s8.steps=v;if(trainer.step===0){s8.total=v;renderTrain();}});
  slider('tr-temp',v=>v.toFixed(1),v=>{s8.temp=v;s8.shown=-1;if(!s8.running)renderTrain();});
  $('#tr-size').addEventListener('change',()=>{newTrainer();renderTrain();gReset();});
  $('#tr-go').addEventListener('click',toggleTrain);
  $('#tr-reset').addEventListener('click',()=>{newTrainer();renderTrain();gReset();});
  newTrainer();reg(()=>{s8.shown=-1;renderTrain();});
}
/* ================= generation and decoding ================= */
const s9={ctx:[],plen:0,steps:[],sel:-1,timer:null,rng:mulberry32(123),o:{temp:0.8,topK:21,topP:1,max:40},dropped:false};
function gStop(){if(s9.timer){clearInterval(s9.timer);s9.timer=null;}}
function gReset(){
  gStop();const raw=$('#g-prompt').value.toLowerCase();let ctx=encode(raw);s9.dropped=ctx.length!==raw.length;
  if(!ctx.length)ctx=encode('the ');
  s9.ctx=ctx;s9.plen=ctx.length;s9.steps=[];s9.sel=-1;s9.rng=mulberry32(123);renderGen();
}
function gStep(){
  if(s9.steps.length>=s9.o.max)return false;
  const d=decodeDist(nextLogits(trainer.model,s9.ctx),s9.o),greedy=$('#g-greedy').checked,top=d.order[0],pick=greedy?top:sampleFrom(d.pF,s9.rng);
  s9.steps.push({d:d,pick:pick,top:top,seen:decode(s9.ctx.slice(-trainer.model.cfg.T)),greedy:greedy});s9.ctx.push(pick);s9.sel=s9.steps.length-1;return true;
}
function renderGenStatus(){
  $('#g-status').innerHTML=(trainer.step===0?'<span class="badge bad">untrained</span> The toy model has random weights, so its output will be gibberish. Train it in <a href="#training">Training an LLM</a>, or press “Train the model now”.':'<span class="badge ok">trained</span> Using the toy model after '+trainer.step+' training steps (validation loss '+fmt(s8.val[s8.val.length-1][1],3)+').'+(s8.running?' Still training…':''))+(s9.dropped?' Characters the model has never seen were removed from your prompt.':'');
  $('#g-train').disabled=s8.running;
}
function renderGen(){
  renderGenStatus();
  const T=trainer.model.cfg.T,n=s9.ctx.length;
  $('#g-context').innerHTML=s9.ctx.map((c,i)=>'<span class="'+(i<s9.plen?'pr':'gen')+(i<n-T?' out':'')+(i===n-1&&i>=s9.plen?' new':'')+'">'+(CHARS[c]===' '?'&nbsp;':CHARS[c])+'</span>').join('')+' <small>('+n+' characters, the model sees the last '+Math.min(n,T)+')</small>';
  $('#g-timeline').innerHTML=s9.steps.length?s9.steps.map((s,i)=>'<button data-i="'+i+'" class="'+(i===s9.sel?'on':'')+'" aria-pressed="'+(i===s9.sel)+'" aria-label="Step '+(i+1)+': '+(CHARS[s.pick]===' '?'space':CHARS[s.pick])+'">'+showSp(CHARS[s.pick])+'<small>'+Math.round(s.d.pF[s.pick]*100)+'%</small></button>').join(''):'<small>Nothing generated yet. Press Generate or Step.</small>';
  $('#g-play').textContent=s9.timer?'⏸ Pause':'▶ Generate';
  if(s9.sel<0){$('#g-dist-lab').textContent='Distribution at the selected step';$('#g-dist').innerHTML='';return;}
  const s=s9.steps[s9.sel],d=s.d,show=d.order.slice(0,10);if(show.indexOf(s.pick)<0)show.push(s.pick);
  $('#g-dist-lab').innerHTML='Step '+(s9.sel+1)+': the model saw “'+esc(s.seen).replace(/ /g,'␣')+'” and '+(s.greedy?'greedily chose':'sampled')+' “'+showSp(CHARS[s.pick])+'”';
  $('#g-dist').innerHTML=bars(show.map(i=>({label:(i===s.top?'★ ':'')+'“'+showSp(CHARS[i])+'”',p:d.pT[i],cls:(d.cut[i]?'cut':'')+(i===s.pick?' hit':''),right:(d.cut[i]?'cut by top-'+d.cut[i]:fmt(d.pT[i],3)+' → '+fmt(d.pF[i],3))+(i===s.pick?' ◀':''),tip:'after temperature: '+d.pT[i].toFixed(5)+'<br>after filtering: '+d.pF[i].toFixed(5)})),1);
}
function initGen(){
  slider('g-temp',v=>v.toFixed(1),v=>{s9.o.temp=v;});slider('g-k',v=>v>=21?'off (21)':v,v=>{s9.o.topK=v;});slider('g-p',v=>v>=1?'off (1.00)':v.toFixed(2),v=>{s9.o.topP=v;});slider('g-max',v=>v,v=>{s9.o.max=v;});
  $('#g-preset').addEventListener('change',e=>{$('#g-prompt').value=e.target.value;gReset();});
  $('#g-prompt').addEventListener('input',gReset);
  $('#g-reset').addEventListener('click',gReset);
  $('#g-step').addEventListener('click',()=>{gStop();gStep();renderGen();});
  $('#g-play').addEventListener('click',()=>{if(s9.timer){gStop();renderGen();return;}if(s9.steps.length>=s9.o.max){gReset();}s9.timer=setInterval(()=>{if(!gStep())gStop();renderGen();},260);renderGen();});
  $('#g-timeline').addEventListener('click',e=>{const b=e.target.closest('button');if(b){gStop();s9.sel=+b.dataset.i;renderGen();const n=$('#g-timeline button[data-i="'+s9.sel+'"]');if(n)n.focus();}});
  $('#g-train').addEventListener('click',()=>{if(!s8.running)toggleTrain();});
  gReset();reg(renderGen);
}
/* ================= GPUs and CUDA: will it fit? ================= */
const HW_GPUS=[['RTX 4090',24e9,1008e9,165.2e12],['L4',24e9,300e9,121e12],['A100',80e9,2039e9,312e12],['H100 SXM',80e9,3350e9,989.5e12],['DGX Spark',128e9,273e9,Infinity]]; /* DGX Spark: no published BF16 figure, so no compute limit */
const sh={m:2,prec:2,g:0,n:1,T:4096,B:1};
function renderHW(){
  const M=IE_MODELS[sh.m],N=M[1],L=M[2],d=M[3],G=HW_GPUS[sh.g],W=N*sh.prec,kv=2*L*sh.T*d*2*sh.B,extra=0.2*(W+kv),need=W+kv+extra,cap=sh.n*G[1],fits=need<=cap,span=Math.max(need,cap);
  const seg=(cls,v,label,tip)=>'<span class="'+cls+'" style="width:'+100*v/span+'%" tabindex="0" data-tip="'+esc(tip+': '+bytes(v))+'">'+(v/span>0.12?label:'')+'</span>';
  $('#hw-bar').innerHTML=seg('p0',W,'weights','Weights')+seg('p2',kv,'KV cache','KV cache')+seg('p4',extra,'+20%','Allowance for the CUDA context, activations and fragmentation')+(fits?'<span style="width:'+100*(cap-need)/span+'%;color:var(--muted)">'+((cap-need)/span>0.12?'free':'')+'</span>':'');
  const fewest=Math.ceil(need/G[1]),gpus=sh.n+' × '+G[0];
  $('#hw-fit').innerHTML='<span class="badge '+(fits?'ok':'bad')+'">'+(fits?'fits':'does not fit')+'</span> Needs '+bytes(need)+'; '+gpus+' '+(sh.n>1?'have':'has')+' '+bytes(cap)+'.'+(fits?'':' The fewest '+G[0]+' GPUs that fit: '+fewest+(fewest>8?', more than one server holds':'')+'.')
    +'<br>1.2 × (weights '+bytes(W)+' + KV cache '+bytes(kv)+') = '+bytes(need)+'<br>KV cache = 2 × '+L+' layers × '+sh.T.toLocaleString('en-US')+' tokens × '+d+' × 2 bytes × '+sh.B+' request'+(sh.B>1?'s':'')+' = '+kv.toLocaleString('en-US')+' bytes';
  const tBw=(W+kv)/(sh.n*G[2]),tFl=2*N*sh.B/(sh.n*G[3]),step=Math.max(tBw,tFl),per=1/step;
  const rate=x=>x>=100?Math.round(x).toLocaleString('en-US'):x>=10?x.toFixed(1):x.toFixed(2);
  $('#hw-stats').innerHTML=[['Weights',bytes(W)],['KV cache',bytes(kv)],['Needed',bytes(need)],['Available',bytes(cap)],['Speed per request',fits?'≤ '+rate(per)+' tok/s':'–'],['Total throughput',fits?'≤ '+rate(per*sh.B)+' tok/s':'–'],['Limited by',fits?(tBw>=tFl?'bandwidth':'compute'):'memory'],['To train it',bytes(16*N)+' + activations']].map(x=>'<div><span>'+x[0]+'</span><b style="font-size:.95rem">'+x[1]+'</b></div>').join('');
  /* total throughput and speed per request for 1 to 64 requests at once, from the same two limits */
  const at=b=>{const k=2*L*sh.T*d*2*b,s=Math.max((W+k)/(sh.n*G[2]),2*N*b/(sh.n*G[3]));return [b/s,1/s];};
  const A=[],Bp=[];let lo=Infinity,hi=-Infinity;
  for(let x=0;x<=6;x++){const [t,p]=at(Math.pow(2,x));A.push([x,Math.log10(t)]);Bp.push([x,Math.log10(p)]);lo=Math.min(lo,Math.log10(p));hi=Math.max(hi,Math.log10(t));}
  lo=Math.floor(lo-0.01);hi=Math.ceil(hi+0.01);if(hi-lo<2)hi=lo+2;
  const yt=[];for(let y=lo;y<=hi;y++)yt.push(y);const x0=Math.log2(sh.B);
  lineChart($('#hw-chart'),[{pts:A,cls:'a'},{pts:Bp,cls:'b'}],{x0:0,x1:6,y0:lo,y1:hi,h:250,ml:54,xt:[0,1,2,3,4,5,6],xf:v=>Math.pow(2,v),yt:yt,yf:v=>v>=0&&v<4?Math.pow(10,v):pow10(v),xl:'requests at once',yl:'tokens per second',dots:[[x0,Math.log10(per*sh.B),sh.B>1?'total':''],[x0,Math.log10(per),sh.B>1?'per request':'']]});
}
function initHW(){
  $('#hw-model').innerHTML=IE_MODELS.map((m,i)=>'<option value="'+i+'"'+(i===sh.m?' selected':'')+'>'+m[0]+'</option>').join('');
  [['hw-model','m'],['hw-prec','prec'],['hw-gpu','g'],['hw-n','n']].forEach(([id,k])=>$('#'+id).addEventListener('change',e=>{sh[k]=+e.target.value;renderHW();}));
  slider('hw-T',v=>Math.pow(2,v).toLocaleString('en-US'),v=>{sh.T=Math.pow(2,v);renderHW();});
  slider('hw-B',v=>Math.pow(2,v),v=>{sh.B=Math.pow(2,v);renderHW();});
  reg(renderHW);
}
/* ================= inference in practice ================= */
const KC_WORDS='the cat sat on the mat and then it went to sleep by the warm fire for the night .'.split(' ');
const sk={P:5,n:6,cache:true,step:6,timer:null};
function kcStop(){if(sk.timer){clearInterval(sk.timer);sk.timer=null;}}
function renderKC(){
  const P=sk.P,n=sk.n,st=Math.min(sk.step,n),tot=P+n,w=KC_WORDS;
  let h='<div class="kc" style="grid-template-columns:auto repeat('+tot+',minmax(26px,1fr))"><span></span>'+w.slice(0,tot).map((x,j)=>'<span class="kh'+(j<P?' pr':'')+'">'+esc(x.slice(0,3))+'</span>').join(''),done=0;
  for(let k=0;k<st;k++){
    done+=sk.cache?(k===0?P:1):P+k;
    h+='<span class="kr">'+(k===0?'prefill':'step '+k)+'</span>';
    for(let j=0;j<tot;j++){
      let cls='',txt='',tip='';
      if(j<P+k){const fresh=!sk.cache||k===0||j===P+k-1;cls=fresh?'cmp':'reu';txt=fresh?'●':'↺';tip='“'+w[j]+'”: keys and values '+(fresh?'computed in this pass':'read from the cache');}
      else if(j===P+k){cls='outc';txt='★';tip='“'+w[j]+'” is the token this pass predicts';}
      h+='<span class="kcc '+cls+'"'+(tip?' data-tip="'+esc(tip)+'"':'')+'>'+txt+'</span>';
    }
  }
  $('#kc-grid').innerHTML=h+'</div>';
  const no=n*P+n*(n-1)/2,yes=P+n-1,k=st-1;
  $('#kc-cap').innerHTML=st===0?'Nothing has run yet. Press <b>Step</b> or <b>Play</b>.':'<b>'+(k===0?'Prefill':'Decode step '+k)+'.</b> The model needs all '+(P+k)+' tokens so far to predict “'+w[P+k]+'”. '+(sk.cache?(k===0?'The cache is empty, so all '+P+' prompt positions are computed and stored.':'Only the newest token, “'+w[P+k-1]+'”, is computed; the other '+(P+k-1)+' are read from the cache.'):'There is no cache, so all '+(P+k)+' positions are computed again from scratch.');
  $('#kc-stats').innerHTML=[['Passes run',st+' / '+n],['Positions computed so far',done],['Tokens held in the cache',sk.cache&&st?P+st-1:0],['Saving over the full run',(no/yes).toFixed(1)+'×']].map(r=>'<div><span>'+r[0]+'</span><b>'+r[1]+'</b></div>').join('');
  $('#kc-bars').innerHTML=bars([{label:'without cache',p:no,right:no+' positions',cls:sk.cache?'':'hit',tip:'n·P + n(n−1)/2 = '+n+'·'+P+' + '+n+'·'+(n-1)+'/2 = '+no},{label:'with cache',p:yes,right:yes+' positions',cls:sk.cache?'hit':'',tip:'P + (n−1) = '+P+' + '+(n-1)+' = '+yes}],no);
  $('#kc-play').textContent=sk.timer?'⏸ Pause':'▶ Play';
}
const IE_MODELS=[['GPT-2 small size · 124M · 12 layers, width 768',124.4e6,12,768],['1B class · 24 layers, width 2048',1.2e9,24,2048],['7B class · 32 layers, width 4096',7e9,32,4096],['70B class · 80 layers, width 8192',7e10,80,8192]];
function initInfer(){
  const reset=()=>{kcStop();sk.step=sk.n;renderKC();};
  slider('kc-P',v=>v,v=>{sk.P=v;reset();});slider('kc-n',v=>v,v=>{sk.n=v;reset();});
  $('#kc-cache').addEventListener('change',e=>{sk.cache=e.target.checked;renderKC();});
  $('#kc-step').addEventListener('click',()=>{kcStop();sk.step=sk.step>=sk.n?0:sk.step+1;renderKC();});
  $('#kc-reset').addEventListener('click',()=>{kcStop();sk.step=0;renderKC();});
  $('#kc-play').addEventListener('click',()=>{if(sk.timer){kcStop();renderKC();return;}if(sk.step>=sk.n)sk.step=0;sk.timer=setInterval(()=>{if(sk.step>=sk.n)kcStop();else sk.step++;renderKC();},700);renderKC();});
  reg(renderKC);
}
/* ================= Post-training: chat template and loss mask ================= */
const CT_MSGS=[['system','You are a helpful assistant.'],['user','What did the cat sit on?'],['assistant','The cat sat on the mat.'],['user','And the dog?'],['assistant','The text does not mention a dog.']];
const sct={turns:1};
function renderChat(){
  const sys=$('#ct-sys').checked,all=$('#ct-all').checked,msgs=CT_MSGS.filter((m,i)=>(i>0||sys)&&i<=2*sct.turns);
  let h='',n=0,tr=0;
  const tok=(t,cls)=>{n++;if(cls==='tk-tr')tr++;h+='<span class="'+cls+'">'+esc(t)+'</span>';};
  msgs.forEach(([role,text])=>{const reply=role==='assistant',cx=all||reply?'tk-tr':'tk-cx',sp=all?'tk-tr':'tk-sp';
    tok('<|im_start|>',sp);tok(role,sp);tok('↵',sp);h+='<i class="nl"></i>';
    (text.match(/\w+|[^\w\s]/g)||[]).forEach(w=>tok(w,cx));tok('<|im_end|>',reply||all?'tk-tr':'tk-sp');tok('↵',sp);h+='<i class="nl"></i>';});
  $('#ct-seq').innerHTML=h;
  $('#ct-turns').innerHTML=[1,2].map(t=>'<button class="chip'+(t===sct.turns?' on':'')+'" data-t="'+t+'" aria-pressed="'+(t===sct.turns)+'">'+t+'</button>').join('');
  $('#ct-stats').innerHTML='<div><b>'+n+'</b><span>tokens in the sequence</span></div><div><b>'+tr+'</b><span>tokens with loss</span></div><div><b>'+Math.round(100*tr/n)+'%</b><span>of the sequence trained</span></div>';
}
function initChat(){$('#ct-turns').addEventListener('click',e=>{const b=e.target.closest('button');if(b){sct.turns=+b.dataset.t;renderChat();$('#ct-turns button[data-t="'+sct.turns+'"]').focus();}});['ct-sys','ct-all'].forEach(id=>$('#'+id).addEventListener('change',renderChat));reg(renderChat);}
/* ================= Post-training: preference tuning on three answers ================= */
const PF_ANS=['A · “The mat.”','B · long, flattering','C · “The sofa.”'],PF_REF=[0.5,0.3,0.2],PF_RM={careful:['Careful',[2,1,-2]],flawed:['Flawed: likes length',[1,3,-2]]};
const spf={rm:'careful',lam:1};
function renderPref(){
  const r=PF_RM[spf.rm][1],w=PF_REF.map((p,i)=>p*Math.exp(r[i]/spf.lam)),Z=w.reduce((a,b)=>a+b),pi=w.map(x=>x/Z);
  const kl=pi.reduce((a,p,i)=>a+(p>0?p*Math.log(p/PF_REF[i]):0),0),er=pi.reduce((a,p,i)=>a+p*r[i],0),best=pi.indexOf(Math.max.apply(null,pi));
  const rows=[];PF_ANS.forEach((a,i)=>{rows.push({label:a+' before',p:PF_REF[i],cls:'cut',right:fmt(PF_REF[i],3)});rows.push({label:a+' tuned',p:pi[i],cls:i===best?'hit':'',right:fmt(pi[i],3)});});
  $('#pf-bars').innerHTML=bars(rows,1);
  $('#pf-rm').innerHTML=Object.keys(PF_RM).map(k=>'<button class="chip'+(k===spf.rm?' on':'')+'" data-k="'+k+'" aria-pressed="'+(k===spf.rm)+'">'+PF_RM[k][0]+'</button>').join('');
  $('#pf-stats').innerHTML='<div><b>'+r.join(', ')+'</b><span>rewards for A, B, C</span></div><div><b>'+fmt(er,2)+'</b><span>expected reward (before: '+fmt(PF_REF.reduce((a,p,i)=>a+p*r[i],0),2)+')</span></div><div><b>'+fmt(kl,3)+'</b><span>KL from the reference, nats</span></div>';
}
function initPref(){
  slider('pf-lam',v=>'λ = '+fmt(Math.pow(10,v),2),v=>{spf.lam=Math.pow(10,v);renderPref();});
  $('#pf-rm').addEventListener('click',e=>{const b=e.target.closest('button');if(b){spf.rm=b.dataset.k;renderPref();$('#pf-rm button[data-k="'+spf.rm+'"]').focus();}});
  reg(renderPref);
}

/* ================= Quantization: round weights to fewer bits ================= */
/* 64 random weights, symmetric round-to-nearest with one scale per group; index 37 can be made an outlier. */
const QZ_W=(()=>{const r=mulberry32(21);return Array.from({length:64},()=>0.05*randn(r));})(),QZ_OUT=37,QZ_Y=0.2;
const sqz={bits:4,g:64,out:false};
function qzRun(){
  const w=QZ_W.slice();if(sqz.out)w[QZ_OUT]=0.6;
  const qm=2**(sqz.bits-1)-1,groups=[],q=[];
  for(let s=0;s<w.length;s+=sqz.g){const d=Math.max(...w.slice(s,s+sqz.g).map(Math.abs))/qm;groups.push({s,d});for(let i=s;i<s+sqz.g;i++)q.push(Math.max(-qm,Math.min(qm,Math.round(w[i]/d))));}
  return{w,qm,groups,q,hat:q.map((v,i)=>v*groups[Math.floor(i/sqz.g)].d)};
}
function renderQuant(){
  const R=qzRun(),X=i=>40+i*9,Y=v=>(130-Math.max(-QZ_Y,Math.min(QZ_Y,v))/QZ_Y*110).toFixed(1);
  let g='<line class="qzax" x1="34" y1="130" x2="616" y2="130"/>'+[QZ_Y,QZ_Y/2,0,-QZ_Y/2,-QZ_Y].map(v=>'<text class="qzt" x="30" y="'+(+Y(v)+4)+'" text-anchor="end">'+v.toFixed(2)+'</text>').join('');
  R.groups.forEach(G=>{
    const x0=X(G.s)-4,x1=X(G.s+sqz.g-1)+4,lv=Math.floor(QZ_Y/G.d);
    if(lv<=40)for(let k=-Math.min(lv,R.qm);k<=Math.min(lv,R.qm);k++)g+='<line class="qzl" x1="'+x0+'" y1="'+Y(k*G.d)+'" x2="'+x1+'" y2="'+Y(k*G.d)+'"/>';
    if(G.s)g+='<line class="qzb" x1="'+(X(G.s)-4.5)+'" y1="14" x2="'+(X(G.s)-4.5)+'" y2="246"/>';
  });
  R.w.forEach((v,i)=>{const big=Math.abs(v)>QZ_Y;
    g+='<line class="qze" x1="'+X(i)+'" y1="'+Y(v)+'" x2="'+X(i)+'" y2="'+Y(R.hat[i])+'"/><circle class="qzo" cx="'+X(i)+'" cy="'+Y(v)+'" r="3.2"/><circle class="qzq" cx="'+X(i)+'" cy="'+Y(R.hat[i])+'" r="2.4"/>';
    if(big)g+='<text class="qzt" x="'+(X(i)+6)+'" y="'+(+Y(v)+4)+'">'+v.toFixed(2)+' ↑ (off the chart)</text>';});
  $('#qz-svg').innerHTML=g;
  const ord=R.w.map((v,i)=>i).filter(i=>!(sqz.out&&i===QZ_OUT)),rms=a=>Math.sqrt(a.reduce((s,v)=>s+v*v,0)/a.length);
  const err=rms(ord.map(i=>R.w[i]-R.hat[i]))/rms(ord.map(i=>R.w[i])),zeros=R.q.filter(v=>v===0).length,bpw=sqz.bits+16/sqz.g;
  $('#qz-stats').innerHTML=[['Bits per weight, with scales',fmt(bpw,2)],['Error on ordinary weights',(100*err).toFixed(1)+'%'],['Weights rounded to zero',zeros+' of 64'],['An 8B model’s weights',bytes(8.03e9*bpw/8)]].map(r=>'<div><b>'+r[1]+'</b><span>'+r[0]+'</span></div>').join('');
  const G0=R.groups[Math.floor((sqz.out?QZ_OUT:0)/sqz.g)],i0=G0.s;
  $('#qz-read').innerHTML=(sqz.out?'The outlier’s group':'Group 1')+': largest weight '+fmt(R.qm*G0.d,3)+', scale = '+fmt(R.qm*G0.d,3)+' ÷ '+R.qm+' = '+fmt(G0.d,4)+'.<br>First weights: '+[0,1,2,3].map(k=>fmt(R.w[i0+k],3)+' → '+R.q[i0+k]+' → '+fmt(R.hat[i0+k],3)).join(' · ');
  [['qz-bits',String(sqz.bits)],['qz-g',String(sqz.g)],['qz-out',sqz.out?'on':'off']].forEach(([id,v])=>$$('#'+id+' button').forEach(b=>{const o=b.dataset.v===v;b.classList.toggle('on',o);b.setAttribute('aria-pressed',o);}));
}
function initQuant(){
  const chips=(id,opts,set)=>{const el=$('#'+id);el.innerHTML=opts.map(o=>'<button class="chip" data-v="'+o[0]+'" aria-pressed="false">'+o[1]+'</button>').join('');
    el.addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;set(b.dataset.v);renderQuant();$('#'+id+' button[data-v="'+b.dataset.v+'"]').focus();});};
  chips('qz-bits',[['8','8'],['4','4'],['3','3'],['2','2']],v=>{sqz.bits=+v;});
  chips('qz-g',[['64','64 (one scale)'],['32','32'],['16','16']],v=>{sqz.g=+v;});
  chips('qz-out',[['off','None'],['on','One weight of 0.6']],v=>{sqz.out=v==='on';});
  reg(renderQuant);
}

/* ================= FlashAttention: attention, one tile at a time ================= */
/* 8 tokens with 2-number queries and keys and 1-number values, causal mask. The tiled path keeps the running
   maximum, sum and weighted sum per query (online softmax); the plain path stores the full grids. */
const FL_T=8,FL=(()=>{const r=mulberry32(31),v=()=>[1.3*randn(r),1.3*randn(r)],Q=[],K=[],V=[];
  for(let i=0;i<FL_T;i++){Q.push(v());K.push(v());V.push(1+Math.floor(r()*9));}
  return{S:Q.map(q=>K.map(k=>(q[0]*k[0]+q[1]*k[1])/Math.SQRT2)),V};})();
const sfl={mode:'tiled',B:4,row:7,step:0,T:32768};
const flTiles=()=>{const t=[],n=FL_T/sfl.B;for(let qb=0;qb<n;qb++)for(let kb=0;kb<=qb;kb++)t.push([qb,kb]);return t;};
const flTotal=()=>sfl.mode==='tiled'?flTiles().length:3;
function flExact(i){const s=FL.S[i].slice(0,i+1),m=Math.max(...s),e=s.map(x=>Math.exp(x-m)),z=e.reduce((a,b)=>a+b,0);return{w:e.map(x=>x/z),c:e.reduce((a,x,j)=>a+x*FL.V[j],0)/z};}
/* the running totals of query i after the tiles processed so far */
function flRun(i){
  let m=-Infinity,r=0,o=0;const hist=[],B=sfl.B;
  flTiles().slice(0,sfl.step).forEach(([qb,kb])=>{
    if(qb!==Math.floor(i/B))return;
    const js=[];for(let j=kb*B;j<(kb+1)*B&&j<=i;j++)js.push(j);
    const m2=Math.max(m,...js.map(j=>FL.S[i][j])),sc=Math.exp(m-m2);
    r=r*sc+js.reduce((a,j)=>a+Math.exp(FL.S[i][j]-m2),0);
    o=o*sc+js.reduce((a,j)=>a+Math.exp(FL.S[i][j]-m2)*FL.V[j],0);
    hist.push({js,m:m2,r,o,rescaled:isFinite(m)&&m2>m,last:kb===qb});m=m2;});
  return hist;
}
function renderFlash(){
  const B=sfl.B,T=FL_T,tiled=sfl.mode==='tiled',tiles=flTiles(),cur=tiled&&sfl.step?tiles[sfl.step-1]:null;
  let h='<div class="flg" style="grid-template-columns:auto repeat('+T+',minmax(34px,1fr))"><span></span>';
  for(let j=0;j<T;j++)h+='<span class="flh">key '+(j+1)+'<br>v = '+FL.V[j]+'</span>';
  for(let i=0;i<T;i++){
    h+='<button type="button" class="flrb'+(i===sfl.row?' on':'')+'" data-i="'+i+'" aria-pressed="'+(i===sfl.row)+'">query '+(i+1)+'</button>';
    for(let j=0;j<T;j++){
      const ti=tiles.findIndex(t=>t[0]===Math.floor(i/B)&&t[1]===Math.floor(j/B));
      let st='';if(j>i)st='msk';else if(tiled){if(ti>=0&&ti<sfl.step-1)st='used';else if(ti===sfl.step-1)st='now';}else if(sfl.step>=1)st='hbm';
      const shown=st&&st!=='msk',val=!tiled&&sfl.step>=2?fmt(flExact(i).w[j],2):fmt(FL.S[i][j],1);
      h+='<span class="flc fls '+st+(tiled&&j%B===0&&j?' tl':'')+(tiled&&i%B===0&&i?' tt':'')+(i===sfl.row?' fol':'')+'">'+(shown?val:'')+'</span>';}
  }
  $('#fl-grid').innerHTML=h+'</div>';
  const n=flTotal(),vis=T*(T+1)/2;
  $('#fl-cap').innerHTML=!sfl.step?'Press Step to begin. Each cell will hold the score of a query against a key; a value <i>v</i> sits above each key.'
    :tiled?'Tile '+sfl.step+' of '+n+': queries '+(cur[0]*B+1)+'–'+(cur[0]*B+B)+' against keys '+(cur[1]*B+1)+'–'+(cur[1]*B+B)+'. Its scores are computed in fast memory, folded into each query’s running totals, and discarded.'+(sfl.step===n?' Done: tiles wholly above the diagonal were never computed.':'')
    :['','All '+vis+' visible scores are computed and written to main memory.','The scores are read back, each row goes through softmax, and '+vis+' probabilities are written to main memory (shown in the cells).','The probabilities and the values are read back and the outputs are written. Done.'][sfl.step];
  const i=sfl.row,ex=flExact(i);
  $('#fl-rlab').textContent='Query '+(i+1)+(tiled?': running totals after each of its tiles':': plain attention');
  let t;
  if(tiled){const hist=flRun(i);
    t='<thead><tr><th>Tile (keys)</th><th>Largest score <i>m</i></th><th>Sum <i>r</i></th><th>Weighted sum <i>o</i></th><th></th></tr></thead><tbody>'+
      (hist.length?hist.map(e=>'<tr><td>'+(e.js[0]+1)+'–'+(e.js[e.js.length-1]+1)+'</td><td>'+fmt(e.m,2)+'</td><td>'+fmt(e.r,3)+'</td><td>'+fmt(e.o,3)+'</td><td>'+(e.rescaled?'<span class="badge bad">largest rose: old totals rescaled</span>':'')+'</td></tr>').join(''):'<tr><td colspan="5">No tile of this query has been processed yet.</td></tr>');
    const done=hist.length&&hist[hist.length-1].last;
    t+='</tbody>'+(done?'<tfoot><tr><th scope="row">Output</th><td colspan="4"><i>o</i> ÷ <i>r</i> = '+fmt(hist[hist.length-1].o/hist[hist.length-1].r,4)+' · exact softmax: '+fmt(ex.c,4)+' <span class="badge ok">same</span></td></tr></tfoot>':'');}
  else t='<tbody><tr><th scope="row">Probabilities</th><td>'+(sfl.step>=2?ex.w.map(w=>fmt(w,3)).join(', '):'not computed yet')+'</td></tr><tr><th scope="row">Output</th><td>'+(sfl.step>=3?fmt(ex.c,4):'not computed yet')+'</td></tr></tbody>';
  $('#fl-run').innerHTML=t;
  const written=tiled?0:(sfl.step>=2?2*vis:sfl.step>=1?vis:0),finished=tiled?tiles.slice(0,sfl.step).filter(x=>x[0]===x[1]).length*B:(sfl.step>=3?T:0);
  $('#fl-stats').innerHTML=[['Step',sfl.step+' / '+n],['Grid values written to main memory',written],['Most scores held at once',tiled?B*B:vis],['Outputs finished',finished+' of '+T]].map(r=>'<div><b>'+r[1]+'</b><span>'+r[0]+'</span></div>').join('');
  const g=sfl.T*sfl.T*32*2;
  $('#fl-mem').innerHTML='At '+sfl.T.toLocaleString('en-US')+' tokens, plain attention stores '+bytes(g)+' of scores per layer for 32 heads, and the same again for the probabilities. FlashAttention holds tiles of 128 × 128 scores, '+bytes(128*128*2)+' each, in on-chip memory.';
  $('#fl-step').disabled=sfl.step>=n;$('#fl-all').disabled=sfl.step>=n;
  [['fl-mode',sfl.mode],['fl-b',String(B)]].forEach(([id,v])=>$$('#'+id+' button').forEach(b=>{const o=b.dataset.v===v;b.classList.toggle('on',o);b.setAttribute('aria-pressed',o);}));
}
function initFlash(){
  const chips=(id,opts,set)=>{const el=$('#'+id);el.innerHTML=opts.map(o=>'<button class="chip" data-v="'+o[0]+'" aria-pressed="false">'+o[1]+'</button>').join('');
    el.addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;set(b.dataset.v);sfl.step=0;renderFlash();$('#'+id+' button[data-v="'+b.dataset.v+'"]').focus();});};
  chips('fl-mode',[['tiled','Tiled (FlashAttention)'],['plain','Plain']],v=>{sfl.mode=v;});
  chips('fl-b',[['2','2 × 2'],['4','4 × 4']],v=>{sfl.B=+v;});
  $('#fl-step').addEventListener('click',()=>{if(sfl.step<flTotal())sfl.step++;renderFlash();});
  $('#fl-all').addEventListener('click',()=>{sfl.step=flTotal();renderFlash();});
  $('#fl-reset').addEventListener('click',()=>{sfl.step=0;renderFlash();});
  $('#fl-grid').addEventListener('click',e=>{const b=e.target.closest('.flrb');if(b){sfl.row=+b.dataset.i;renderFlash();$('#fl-grid .flrb[data-i="'+sfl.row+'"]').focus();}});
  slider('fl-T',v=>Math.pow(2,v).toLocaleString('en-US'),v=>{sfl.T=Math.pow(2,v);renderFlash();});
  reg(renderFlash);
}

PAGE_INITS.push(initTrain,initGen,initHW,initInfer,initChat,initPref,initQuant,initFlash);
