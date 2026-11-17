'use strict';
/* Part B widgets: tokens, positions, attention, multi-head attention, the block and the architecture. */
function pe(pos,i,d){const k=Math.floor(i/2),ang=pos/Math.pow(10000,2*k/d);return i%2?Math.cos(ang):Math.sin(ang);}
/* ---------- toy WordPiece-style tokenizer ---------- */
const TOK=(function(){
  const words=['the','cat','dog','mouse','bird','horse','man','woman','king','queen','prince','sat','ran','walk','jump','play','bite','bites','eat','sleep','on','in','under','with','and','mat','log','tree','house','happy','sad','big','small','un','re','quick','slow','love','see'];
  const suff=['s','ed','ing','er','ly','est','ness','happy','able'];
  const letters='abcdefghijklmnopqrstuvwxyz'.split('');
  const vocab=['[UNK]','.',',','!','?'].concat(words,letters,suff.filter(s=>s.length>1).map(s=>'##'+s),letters.map(l=>'##'+l));
  const id=new Map(vocab.map((t,i)=>[t,i]));
  function word(w){const out=[];let p=0;while(p<w.length){for(let e=w.length;e>p;e--){const piece=(p?'##':'')+w.slice(p,e);if(id.has(piece)){out.push(piece);p=e;break;}}}return out;}
  function tokenize(text){const parts=text.toLowerCase().match(/[a-z]+|[^a-z\s]/g)||[];const out=[];for(const part of parts){if(/^[a-z]+$/.test(part))out.push.apply(out,word(part));else out.push(id.has(part)?part:'[UNK]');}return out.map(t=>({t:t,id:id.get(t)}));}
  const rng=mulberry32(7),d=4;
  const E=vocab.map(()=>Array.from({length:d},()=>Math.round(randn(rng)*50)/100));
  return {vocab,id,tokenize,E,d};
})();
const SCATTER=[
  ['cat',2.1,2.3,'animal'],['dog',2.5,1.9,'animal'],['mouse',1.6,2.8,'animal'],['bird',3.0,2.9,'animal'],['horse',3.1,1.5,'animal'],
  ['man',-2.2,1.9,'person'],['woman',-2.7,2.4,'person'],['king',-1.5,3.0,'person'],['queen',-2.0,3.5,'person'],['prince',-0.5,2.1,'person'],['princess',-1.0,2.6,'person'],
  ['sat',-2.0,-1.6,'verb'],['ran',-2.8,-2.1,'verb'],['walk',-2.4,-2.9,'verb'],['jump',-1.5,-2.6,'verb'],['play',-1.2,-1.8,'verb'],['eat',-3.2,-1.4,'verb'],
  ['the',2.2,-1.8,'function'],['on',1.5,-2.5,'function'],['in',2.0,-3.0,'function'],['under',1.1,-1.9,'function'],['with',2.8,-2.6,'function'],['and',3.1,-1.7,'function'],
  ['happy',0.4,0.7,'adjective'],['sad',-0.4,0.3,'adjective'],['big',0.6,-0.3,'adjective'],['small',-0.2,-0.6,'adjective']
].map(a=>({t:a[0],x:a[1],y:a[2],c:a[3]}));
const ORD={dog:[1.0,0.2,-0.5,0.3],bites:[-0.4,0.9,0.1,-0.7],man:[0.6,-0.3,0.8,0.2]};
/* ---------- hand-picked Q, K, V so the arithmetic is checkable ---------- */
const AT={
  tok:['the','cat','sat','on','mat'],
  Q:[[1,0,1,0],[0,1,0,1],[1,2,0,0],[0,0,1,1],[0,1,2,0]],
  K:[[1,0,0,1],[0,2,1,0],[1,1,0,-1],[0,0,2,1],[-1,1,1,0]],
  V:[[0.5,0,0,1],[1,2,0,0],[0,1,1,0],[0,0,1,2],[2,0,1,1]]
};
/* ---------- hand-designed heads ---------- */
const MH=(function(){
  const tok=['the','cat','sat','because','it','purred'],n=6;
  const z=()=>Array.from({length:n},()=>new Array(n).fill(0));
  const prev=z();for(let i=0;i<n;i++)prev[i][Math.max(0,i-1)]=4;
  const coref=z();for(let i=0;i<n;i++)coref[i][i]=1.5;coref[4][4]=0;coref[4][1]=4;
  const subj=z();for(let i=0;i<n;i++)subj[i][i]=1;subj[2][1]=4;subj[5][4]=4;subj[5][1]=2;
  const broad=z();
  const base=[{name:'Previous token',note:'adjacency: each token looks one step back',S:prev},{name:'Coreference',note:'“it” looks back at “cat”',S:coref},{name:'Subject finder',note:'syntax: each verb looks for its subject',S:subj},{name:'Broad context',note:'long range: averages everything visible',S:broad}];
  const rng=mulberry32(11);
  const V=Array.from({length:n},()=>Array.from({length:8},()=>Math.round(randn(rng)*10)/10));
  function heads(h){
    const groups=h===4?[[0],[1],[2],[3]]:h===2?[[0,1],[2,3]]:[[0,1,2,3]],dh=8/h;
    return groups.map((g,k)=>{
      const S=z();for(let i=0;i<n;i++)for(let j=0;j<n;j++){let s=0;for(const b of g)s+=base[b].S[i][j];S[i][j]=s/g.length;}
      const A=S.map((row,i)=>softmax(row.map((s,j)=>j>i?-Infinity:s)));
      const out=A.map(row=>Array.from({length:dh},(_,c)=>row.reduce((s,a,j)=>s+a*V[j][k*dh+c],0)));
      return {name:g.map(b=>base[b].name).join(' + '),note:g.length>1?'one head forced to share '+g.length+' jobs':base[g[0]].note,S,A,out};
    });
  }
  return {tok,V,heads};
})();
/* ---------- signal / gradient propagation through a deep random stack ---------- */
function signalSim(o){
  const rng=mulberry32(o.seed),d=o.d,L=o.L,std=o.gain/Math.sqrt(d);
  const rms=v=>{let s=0;for(const a of v)s+=a*a;return Math.sqrt(s/v.length);};
  const mat=()=>{const w=new Float64Array(d*d);for(let i=0;i<w.length;i++)w[i]=randn(rng)*std;return w;};
  const mv=(W,x)=>{const y=new Float64Array(d);for(let i=0;i<d;i++){let s=0;for(let j=0;j<d;j++)s+=W[i*d+j]*x[j];y[i]=s;}return y;};
  const mvT=(W,x)=>{const y=new Float64Array(d);for(let i=0;i<d;i++){const xi=x[i];for(let j=0;j<d;j++)y[j]+=W[i*d+j]*xi;}return y;};
  let x=new Float64Array(d);for(let i=0;i<d;i++)x[i]=randn(rng);
  const fwd=[rms(x)],cache=[];
  for(let l=0;l<L;l++){
    const W1=mat(),W2=mat();let h=x,sd=1;
    if(o.norm){let mu=0;for(const a of x)mu+=a;mu/=d;let va=0;for(const a of x)va+=(a-mu)*(a-mu);sd=Math.sqrt(va/d+1e-5);h=x.map(a=>(a-mu)/sd);}
    const a=mv(W1,h),f=mv(W2,a.map(gelu)),y=new Float64Array(d);
    for(let i=0;i<d;i++)y[i]=(o.residual?x[i]:0)+f[i];
    cache.push({W1,W2,h,a,sd});x=y;fwd.push(rms(x));
  }
  let g=new Float64Array(d);for(let i=0;i<d;i++)g[i]=randn(rng);const g0=rms(g);for(let i=0;i<d;i++)g[i]/=g0;
  const grad=new Array(L+1);grad[L]=rms(g);
  for(let l=L-1;l>=0;l--){
    const c=cache[l];const dg=mvT(c.W2,g);const da=dg.map((v,i)=>v*geluGrad(c.a[i]));let dx=mvT(c.W1,da);
    if(o.norm){let m1=0,m2=0;for(let i=0;i<d;i++){m1+=dx[i];m2+=dx[i]*c.h[i];}m1/=d;m2/=d;dx=dx.map((v,i)=>(v-m1-c.h[i]*m2)/c.sd);}
    if(o.residual)for(let i=0;i<d;i++)dx[i]+=g[i];
    g=dx;grad[l]=rms(g);
  }
  return {fwd,grad};
}
/* ---------- parameter counting ---------- */
function paramCounts(c){
  const tok=c.V*c.d,pos=c.T*c.d,attn=c.L*4*(c.d*c.d+c.d),mlp=c.L*(2*c.d*c.ff+c.ff+c.d),ln=c.L*4*c.d+2*c.d,out=c.tie?0:c.V*c.d;
  return {tok,pos,attn,mlp,ln,out,total:tok+pos+attn+mlp+ln+out};
}
/* ================= tokenizer and embeddings ================= */
const s2={sel:0,present:[],pt:null,ar:{a:'king',b:'man',c:'woman',ex:true}};
function renderTok(){
  const text=$('#tk-in').value,toks=TOK.tokenize(text);if(s2.sel>=toks.length)s2.sel=0;
  s2.present=toks.map(t=>t.t);
  $('#tk-cmp').innerHTML='The same text is <b>'+text.length+'</b> characters, <b>'+(text.trim().match(/\S+/g)||[]).length+'</b> whitespace-separated words, or <b>'+toks.length+'</b> subword tokens.';
  $('#tk-chips').innerHTML=toks.map((t,i)=>'<button class="chip tok'+(i===s2.sel?' on':'')+'" data-i="'+i+'" aria-pressed="'+(i===s2.sel)+'"><span>'+esc(t.t)+'</span><small>id '+t.id+'</small></button>').join('')||'<em>Type something above.</em>';
  if(!toks.length){$('#tk-detail').innerHTML='';renderScatter();return;}
  const t=toks[s2.sel],V=TOK.vocab.length;
  $('#tk-detail').innerHTML='<span class="lab">Token → ID</span><code>'+esc(t.t)+'</code> is entry <b>'+t.id+'</b> of '+V+' in the vocabulary.'+
    '<span class="lab">One-hot vector (length '+V+')</span><div class="onehot" role="img" aria-label="One-hot vector with a single 1 at index '+t.id+'">'+TOK.vocab.map((_,i)=>'<i'+(i===t.id?' class="on"':'')+'></i>').join('')+'</div><small>'+V+' numbers: a single 1 at index '+t.id+', and 0 everywhere else.</small>'+
    '<span class="lab">Embedding vector x = E['+t.id+'] ('+sym('d_model')+' = 4)</span>'+vec(TOK.E[t.id],'',2,1.5)+' <small>four numbers that training would tune</small>';
  renderScatter();
}
const SHAPES={
  animal:(x,y)=>'<circle class="sh" cx="'+x+'" cy="'+y+'" r="6"/>',
  person:(x,y)=>'<rect class="sh" x="'+(x-5.5)+'" y="'+(y-5.5)+'" width="11" height="11"/>',
  verb:(x,y)=>'<polygon class="sh" points="'+x+','+(y-7)+' '+(x+7)+','+(y+5)+' '+(x-7)+','+(y+5)+'"/>',
  function:(x,y)=>'<polygon class="sh" points="'+x+','+(y-7)+' '+(x+7)+','+y+' '+x+','+(y+7)+' '+(x-7)+','+y+'"/>',
  adjective:(x,y)=>'<path class="sh" d="M'+(x-2)+' '+(y-7)+'h4v5h5v4h-5v5h-4v-5h-5v-4h5z"/>'
};
function renderScatter(){
  const svg=$('#sc-emb'),W=Math.max(260,Math.round(svg.getBoundingClientRect().width)||520),H=340;
  svg.setAttribute('viewBox','0 0 '+W+' '+H);svg.style.height=H+'px';
  const X=x=>20+(x+4.2)/8.4*(W-40),Y=y=>H-16-(y+4)/8.2*(H-32);
  let s='<line class="grid" x1="'+X(0)+'" x2="'+X(0)+'" y1="8" y2="'+(H-8)+'"/><line class="grid" x1="8" x2="'+(W-8)+'" y1="'+Y(0)+'" y2="'+Y(0)+'"/>'+arithSvg(X,Y);
  const sel=SCATTER.find(p=>p.t===s2.pt);let nb=[];
  if(sel){nb=SCATTER.filter(p=>p!==sel).map(p=>({p:p,d:Math.hypot(p.x-sel.x,p.y-sel.y)})).sort((a,b)=>a.d-b.d).slice(0,3);nb.forEach(n=>{s+='<line class="nb" x1="'+X(sel.x)+'" y1="'+Y(sel.y)+'" x2="'+X(n.p.x)+'" y2="'+Y(n.p.y)+'"/>';});}
  SCATTER.forEach(p=>{const x=+X(p.x).toFixed(1),y=+Y(p.y).toFixed(1),pres=s2.present.indexOf(p.t)>=0,right=x>W-70;
    s+='<g class="pt c-'+p.c+(pres?' present':'')+(p===sel?' sel':'')+'" tabindex="0" role="button" aria-label="'+p.t+', '+p.c+'" data-t="'+p.t+'" data-tip="'+esc('<b>'+p.t+'</b> · '+p.c+'<br>('+p.x.toFixed(1)+', '+p.y.toFixed(1)+')'+(pres?'<br>in your sentence':''))+'">'+(pres?'<circle class="ring" cx="'+x+'" cy="'+y+'" r="11"/>':'')+SHAPES[p.c](x,y)+'<text x="'+(right?x-11:x+11)+'" y="'+(y+4)+'" text-anchor="'+(right?'end':'start')+'">'+p.t+'</text></g>';});
  svg.innerHTML=s;
  $('#sc-out').innerHTML=sel?'Nearest to “'+sel.t+'” ('+sel.c+'): '+nb.map(n=>n.p.t+' ('+n.p.c+', distance '+fmt(n.d,2)+')').join(', ')+'.':'Click or focus a token to see its nearest neighbours.';
  renderArith();
}
/* vector arithmetic on the toy embeddings: a − b + c, then rank the nearest tokens */
const SC_PRE=[['king','man','woman'],['prince','man','woman'],['big','small','sad'],['dog','cat','man']];
function scArith(a,b,c,ex){
  const P=t=>SCATTER.find(p=>p.t===t),A=P(a),B=P(b),C=P(c),r={x:A.x-B.x+C.x,y:A.y-B.y+C.y};
  const rank=SCATTER.filter(p=>!ex||[a,b,c].indexOf(p.t)<0).map(p=>({p,d:Math.hypot(p.x-r.x,p.y-r.y)})).sort((u,v)=>u.d-v.d);
  return{A,B,C,r,rank};
}
function renderArith(){
  const s=s2.ar,R=scArith(s.a,s.b,s.c,s.ex),f=v=>fmt(v,1).replace('-','−');
  $('#sc-ar').innerHTML='<p class="readout"><b>'+s.a+'</b> − <b>'+s.b+'</b> + <b>'+s.c+'</b> = ('+f(R.A.x)+', '+f(R.A.y)+') − ('+f(R.B.x)+', '+f(R.B.y)+') + ('+f(R.C.x)+', '+f(R.C.y)+') = <b>('+f(R.r.x)+', '+f(R.r.y)+')</b>. Nearest token'+(s.ex?' other than the three inputs':'')+': <b>'+R.rank[0].p.t+'</b>, at distance '+fmt(R.rank[0].d,2)+'.</p>'+
    '<div class="scroll"><table class="numt"><thead><tr><th>Rank</th><th>Token</th><th>Kind</th><th>Distance to the result</th></tr></thead><tbody>'+R.rank.slice(0,5).map((n,k)=>'<tr'+(k?'':' class="hit"')+'><td>'+(k+1)+'</td><td>'+n.p.t+'</td><td>'+n.p.c+'</td><td>'+fmt(n.d,2)+'</td></tr>').join('')+'</tbody></table></div>';
  $$('#sc-pre button').forEach((bt,k)=>{const on=SC_PRE[k].join()===[s.a,s.b,s.c].join();bt.classList.toggle('on',on);bt.setAttribute('aria-pressed',on);});
  return R;
}
function arithSvg(X,Y){
  const R=scArith(s2.ar.a,s2.ar.b,s2.ar.c,s2.ar.ex),x=X(R.r.x),y=Y(R.r.y),best=R.rank[0].p;
  const ln=(p,q,cls)=>'<line class="'+cls+'" x1="'+X(p.x)+'" y1="'+Y(p.y)+'" x2="'+X(q.x)+'" y2="'+Y(q.y)+'" marker-end="url(#sc-ah)"/>';
  return '<defs><marker id="sc-ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z"/></marker></defs>'+
    ln(R.B,R.A,'arv')+ln(R.C,R.r,'arv')+'<line class="nb" x1="'+x+'" y1="'+y+'" x2="'+X(best.x)+'" y2="'+Y(best.y)+'"/>'+
    '<path class="arp" d="M'+(x-6)+' '+(y-6)+'L'+(x+6)+' '+(y+6)+'M'+(x+6)+' '+(y-6)+'L'+(x-6)+' '+(y+6)+'"/>';
}
function initArith(){
  const opts=SCATTER.map(p=>p.t).slice().sort().map(t=>'<option value="'+t+'">'+t+'</option>').join('');
  ['a','b','c'].forEach(k=>{const el=$('#sc-'+k);el.innerHTML=opts;el.value=s2.ar[k];el.addEventListener('change',()=>{s2.ar[k]=el.value;renderScatter();});});
  $('#sc-pre').innerHTML=SC_PRE.map((p,k)=>'<button class="chip" data-k="'+k+'" aria-pressed="false">'+p[0]+' − '+p[1]+' + '+p[2]+'</button>').join('');
  $('#sc-pre').addEventListener('click',e=>{const bt=e.target.closest('button');if(!bt)return;const p=SC_PRE[+bt.dataset.k];
    ['a','b','c'].forEach((k,i)=>{s2.ar[k]=p[i];const el=$('#sc-'+k);el.value=p[i];el.dispatchEvent(new Event('change'));});bt.focus();});
  $('#sc-ex').addEventListener('change',e=>{s2.ar.ex=e.target.checked;renderScatter();});
}
function initTok(){
  $('#tk-in').addEventListener('input',()=>{s2.sel=0;renderTok();});
  $('#tk-chips').addEventListener('click',e=>{const b=e.target.closest('button');if(b){s2.sel=+b.dataset.i;renderTok();const n=$('#tk-chips button[data-i="'+s2.sel+'"]');if(n)n.focus();}});
  const pick=e=>{const g=e.target.closest('.pt');if(!g)return;if(e.type==='keydown'){if(e.key!=='Enter'&&e.key!==' ')return;e.preventDefault();}s2.pt=g.dataset.t;renderScatter();const n=$('#sc-emb .pt[data-t="'+s2.pt+'"]');if(n&&e.type==='keydown')n.focus();};
  $('#sc-emb').addEventListener('click',pick);$('#sc-emb').addEventListener('keydown',pick);
  $('#sc-lg').innerHTML=Object.keys(SHAPES).map(c=>'<span><svg width="18" height="18" class="pt c-'+c+'">'+SHAPES[c](9,9)+'</svg> '+c+'</span>').join('')+'<span>◯ ring = in your sentence</span>';
  initArith();
  reg(renderTok);
}
/* ================= positional encoding ================= */
const s3={T:24,d:32,cur:[1,0]};
function renderPE(){
  const cv=$('#pe-cv'),W=Math.max(240,Math.round(cv.parentElement.getBoundingClientRect().width-2)||600),ch=Math.max(6,Math.min(22,Math.floor(380/s3.T))),H=ch*s3.T,dpr=window.devicePixelRatio||1;
  s3.cur[0]=Math.min(s3.cur[0],s3.T-1);s3.cur[1]=Math.min(s3.cur[1],s3.d-1);
  cv.width=Math.round(W*dpr);cv.height=Math.round(H*dpr);cv.style.height=H+'px';
  const g=cv.getContext('2d');g.setTransform(dpr,0,0,dpr,0,0);const cw=W/s3.d;
  for(let p=0;p<s3.T;p++)for(let i=0;i<s3.d;i++){g.fillStyle=divColor(pe(p,i,s3.d));g.fillRect(i*cw,p*ch,cw+0.6,ch+0.6);}
  g.strokeStyle='#111';g.lineWidth=2;g.strokeRect(s3.cur[1]*cw+1,s3.cur[0]*ch+1,cw-2,ch-2);g.strokeStyle='#fff';g.lineWidth=1;g.strokeRect(s3.cur[1]*cw+2.5,s3.cur[0]*ch+2.5,cw-5,ch-5);
  const p=s3.cur[0],i=s3.cur[1],k=Math.floor(i/2),wl=2*Math.PI*Math.pow(10000,2*k/s3.d);
  $('#pe-read').textContent='position t = '+p+', dimension '+i+' ('+(i%2?'cos':'sin')+', pair i = '+k+'): PE = '+fmt(pe(p,i,s3.d))+' · this dimension repeats every '+(wl<1000?wl.toFixed(1):sci(wl))+' positions';
  s3.geom={cw:cw,ch:ch};
}
const ORDS={A:['dog','bites','man'],B:['man','bites','dog']};
function renderOrder(){
  const on=$('#or-pe').checked,n=4;
  const vecOf=(w,p)=>ORD[w].map((x,i)=>x+(on?pe(p,i,4):0));
  const block=(name,words)=>'<div><span class="lab">'+name+'</span>'+words.map((w,p)=>'<div style="margin:4px 0"><code>'+w+'</code> <small>slot '+p+'</small><br>'+vec(vecOf(w,p),w==='dog'?'ctx':'',n,2)+'</div>').join('')+'</div>';
  const a=vecOf('dog',0),b=vecOf('dog',2),same=a.every((v,i)=>Math.abs(v-b[i])<1e-12);
  $('#or-out').innerHTML='<div class="grid2">'+block('“dog bites man”',ORDS.A)+block('“man bites dog”',ORDS.B)+'</div><p class="readout">Vector for “dog” in sentence 1: ['+a.map(v=>fmt(v,n)).join(', ')+']<br>Vector for “dog” in sentence 2: ['+b.map(v=>fmt(v,n)).join(', ')+']<br>'+(same?'<span class="badge bad">identical</span> Both sentences are the same set of three vectors. The model cannot tell them apart.':'<span class="badge ok">different</span> Position 0 and position 2 add different encodings, so the two sentences are now distinct inputs.')+'</p>';
}
function initPE(){
  slider('pe-T',v=>v,v=>{s3.T=v;renderPE();});slider('pe-d',v=>v,v=>{s3.d=v;renderPE();});
  const cv=$('#pe-cv');
  const move=e=>{const r=cv.getBoundingClientRect(),pt=e.touches?e.touches[0]:e;s3.cur=[Math.max(0,Math.min(s3.T-1,Math.floor((pt.clientY-r.top)/s3.geom.ch))),Math.max(0,Math.min(s3.d-1,Math.floor((pt.clientX-r.left)/s3.geom.cw)))];renderPE();};
  cv.addEventListener('mousemove',move);cv.addEventListener('click',move);cv.addEventListener('touchstart',move,{passive:true});
  cv.addEventListener('keydown',e=>{const d={ArrowUp:[-1,0],ArrowDown:[1,0],ArrowLeft:[0,-1],ArrowRight:[0,1]}[e.key];if(!d)return;e.preventDefault();s3.cur=[Math.max(0,Math.min(s3.T-1,s3.cur[0]+d[0])),Math.max(0,Math.min(s3.d-1,s3.cur[1]+d[1]))];renderPE();});
  $('#or-pe').addEventListener('change',renderOrder);
  reg(renderPE);reg(renderOrder);
}
/* ================= self-attention ================= */
const s4={q:2,tau:1,mask:false,step:0,timer:null,hov:null};
const AT_STEP=['Look up','Dot products','Scale','Softmax','Weighted sum'];
function attnCalc(i){
  const den=Math.sqrt(4)*s4.tau,dots=AT.K.map(k=>dot(AT.Q[i],k)),scaled=dots.map((s,j)=>s4.mask&&j>i?-Infinity:s/den),exps=scaled.map(Math.exp),w=softmax(scaled);
  return {den:den,dots:dots,scaled:scaled,exps:exps,w:w,ctx:[0,1,2,3].map(c=>w.reduce((s,a,j)=>s+a*AT.V[j][c],0))};
}
function renderAttn(){
  const i=s4.q,c=attnCalc(i),st=s4.step,tok=AT.tok[i];
  $('#at-toks').innerHTML=AT.tok.map((t,j)=>'<button class="chip'+(j===i?' on':'')+'" data-i="'+j+'" aria-pressed="'+(j===i)+'">'+t+'</button>').join('');
  $('#at-steps').innerHTML=AT_STEP.map((t,k)=>'<li><button data-s="'+k+'" class="'+(k===st?'on':k<st?'done':'')+'"'+(k===st?' aria-current="step"':'')+'>'+(k+1)+'. '+t+'</button></li>').join('');
  $('#at-play').textContent=s4.timer?'⏸ Pause':'▶ Play';
  $('#at-cap').innerHTML=[
    '<b>Step 1 · Look up.</b> “'+tok+'” supplies a <b class="tq">query</b>. Every token, including “'+tok+'” itself, supplies a <b class="tk">key</b> and a <b class="tv">value</b>.',
    '<b>Step 2 · Dot products.</b> Multiply the query with each key element by element and add up. A larger score means a better match.',
    '<b>Step 3 · Scale'+(s4.mask?' and mask':'')+'.</b> Divide each score by √d<sub>k</sub> × τ = 2 × '+fmt(s4.tau,2)+' = '+fmt(c.den,2)+'.'+(s4.mask?' Tokens that come after “'+tok+'” are set to −∞.':''),
    '<b>Step 4 · Softmax.</b> Exponentiate each scaled score and divide by the total. The weights are positive and sum to 1.',
    '<b>Step 5 · Weighted sum.</b> Multiply each value vector by its weight and add them up. The result is the context vector for “'+tok+'”.'][st];
  $('#at-q').innerHTML='<span class="lab">Query for “'+tok+'”</span><b class="tq">q</b> = '+vec(AT.Q[i],'q',1,2);
  const cell=(k,h)=>'<td class="'+(st<k?'hid':st===k?'cur':'')+'">'+h+'</td>',hd=(k,h)=>'<th class="'+(st<k?'hid':'')+'">'+h+'</th>';
  let h='<thead><tr><th>Token j</th><th><span class="tk">Key k<sub>j</sub></span></th>'+hd(1,'q · k<sub>j</sub>')+hd(2,'÷ '+fmt(c.den,2))+hd(3,'exp')+hd(3,'Weight α<sub>j</sub>')+'<th><span class="tv">Value v<sub>j</sub></span></th></tr></thead><tbody>';
  AT.tok.forEach((t,j)=>{const mk=s4.mask&&j>i,ex=AT.Q[i].map((a,x)=>a+'·'+AT.K[j][x]).join(' + ').replace(/-/g,'−');
    h+='<tr class="'+(mk&&st>=2?'mk':'')+'"><th scope="row">'+t+(mk?' <small>(future)</small>':'')+'</th><td>'+vec(AT.K[j],'k',1,2)+'</td>'+cell(1,'<b>'+fmt(c.dots[j],0)+'</b>'+'<small class="ex">'+ex+'</small>')+cell(2,fmt(c.scaled[j]))+cell(3,fmt(c.exps[j]))+cell(3,'<b>'+fmt(c.w[j])+'</b>')+'<td>'+vec(AT.V[j],'v',1,2)+'</td></tr>';});
  h+='<tr><th scope="row">Σ</th><td></td>'+cell(9,'')+cell(9,'')+cell(3,fmt(c.exps.reduce((a,b)=>a+b,0)))+cell(3,fmt(c.w.reduce((a,b)=>a+b,0)))+'<td></td></tr>';
  $('#at-table').innerHTML=h+'</tbody>';
  $('#at-soft').innerHTML=st>=3?'<span class="lab">Softmax distribution: how much “'+tok+'” attends to each token</span>'+bars(AT.tok.map((t,j)=>({label:t,p:c.w[j],tip:'α = '+c.w[j].toFixed(6)})),1):'';
  $('#at-ctx').innerHTML=st>=4?'<span class="lab">Context vector c = Σ α<sub>j</sub> v<sub>j</sub></span>'+vec(c.ctx,'ctx',null,2)+'<div class="readout">'+[0,1,2,3].map(d=>'c<sub>'+d+'</sub> = '+AT.tok.map((t,j)=>fmt(c.w[j])+'·'+AT.V[j][d]).join(' + ')+' = <b>'+fmt(c.ctx[d])+'</b>').join('<br>')+'</div>':'';
  renderMat();
}
function renderMat(){
  const A=AT.tok.map((_,i)=>attnCalc(i).w);
  $('#at-mat').innerHTML=heat(A,AT.tok,AT.tok,{click:true,sel:s4.q,mask:(i,j)=>s4.mask&&j>i,tip:(i,j,v,mk)=>'query <b>'+AT.tok[i]+'</b> → key <b>'+AT.tok[j]+'</b><br>weight '+v.toFixed(6)+(mk?'<br>masked: j &gt; i':'')});
  $('#at-maskgrid').innerHTML='<div class="hm maskg" style="grid-template-columns:auto repeat(5,minmax(0,1fr))"><span></span>'+AT.tok.map(t=>'<span class="hh">'+t+'</span>').join('')+AT.tok.map((t,i)=>'<span class="hr">'+t+'</span>'+AT.tok.map((u,j)=>'<span class="hc'+(j>i?' masked':'')+'" data-tip="'+esc('row '+i+' ('+t+'), column '+j+' ('+u+'): '+(j>i?'j &gt; i, add −∞ (blocked)':'j ≤ i, add 0 (allowed)'))+'">'+(j>i?'−∞':'0')+'</span>').join('')).join('')+'</div><small>'+(s4.mask?'Applied. ':'Not applied yet: tick the box. ')+'0 = allowed, −∞ = blocked (hatched).</small>';
  attnLink();
}
/* hover linking: one cell of the matrix ↔ its query row, key column and value */
function attnLink(){
  const c0=attnCalc(s4.q),best=c0.w.indexOf(Math.max(...c0.w)),[i,j]=s4.hov||[s4.q,best],c=attnCalc(i),mk=s4.mask&&j>i,ti=AT.tok[i],tj=AT.tok[j],a=c.w[j];
  $$('#at-mat .hc').forEach(e=>{e.classList.toggle('lrow',+e.dataset.i===i);e.classList.toggle('lcol',+e.dataset.j===j);e.classList.toggle('lcell',+e.dataset.i===i&&+e.dataset.j===j);});
  $$('#at-mat .hr').forEach((e,k)=>e.classList.toggle('lq',k===i));$$('#at-mat .hh').forEach((e,k)=>e.classList.toggle('lk',k===j));
  $$('#at-maskgrid .hc').forEach((e,k)=>e.classList.toggle('lcell',k===i*AT.tok.length+j));
  const ex=AT.Q[i].map((x,d)=>x+'·'+AT.K[j][d]).join(' + ').replace(/-/g,'−');
  const row=(l,h)=>'<span class="atl-l">'+l+'</span><span>'+h+'</span>';
  $('#at-link').innerHTML='<span class="lab">Row <b class="tq">'+ti+'</b> (the query), column <b class="tk">'+tj+'</b> (the key and value)</span><div class="atl">'+
    row('<b class="tq">q</b> for “'+ti+'”',vec(AT.Q[i],'q',1,2))+
    row('<b class="tk">k</b> for “'+tj+'”',vec(AT.K[j],'k',1,2))+
    row('Score',mk?'masked: “'+tj+'” comes after “'+ti+'”, so the score becomes −∞':'q · k = '+ex+' = <b>'+fmt(c.dots[j],0)+'</b>, ÷ '+fmt(c.den,2)+' = <b>'+fmt(c.scaled[j])+'</b>')+
    row('Weight',mk?'<b>0</b>':'softmax over row “'+ti+'” gives <b>'+fmt(a)+'</b>, the number in the cell')+
    row('<b class="tv">v</b> for “'+tj+'”',vec(AT.V[j],'v',1,2))+
    row('Weight × <b class="tv">v</b>',vec(AT.V[j].map(v=>a*v),'v',2,2))+
    row('Output for “'+ti+'”',vec(c.ctx,'ctx',2,2)+' <small>the sum of weight × <b class="tv">v</b> over the row’s '+(s4.mask?i+1:AT.tok.length)+' cells; this cell supplies '+(mk?'nothing':'the line above')+'</small>')+'</div>';
}
function initAttnLink(){
  const m=$('#at-mat'),set=e=>{const h=e.target.closest('.hc');if(!h)return;const v=[+h.dataset.i,+h.dataset.j];if(s4.hov&&s4.hov[0]===v[0]&&s4.hov[1]===v[1]&&m.classList.contains('lnk'))return;s4.hov=v;m.classList.add('lnk');attnLink();};
  m.addEventListener('pointerover',set);m.addEventListener('focusin',set);
  const off=()=>m.classList.remove('lnk');m.addEventListener('pointerleave',off);m.addEventListener('focusout',off);
}
function attnStop(){if(s4.timer){clearInterval(s4.timer);s4.timer=null;}}
function initAttn(){
  $('#at-toks').addEventListener('click',e=>{const b=e.target.closest('button');if(b){s4.q=+b.dataset.i;renderAttn();}});
  $('#at-steps').addEventListener('click',e=>{const b=e.target.closest('button');if(b){attnStop();s4.step=+b.dataset.s;renderAttn();}});
  slider('at-tau',v=>Math.pow(2,v).toFixed(2),v=>{s4.tau=Math.pow(2,v);if(s4.step<3){attnStop();s4.step=3;}renderAttn();});
  const setMask=v=>{s4.mask=v;$('#at-mask').checked=v;$('#at-mask2').checked=v;renderAttn();};
  $('#at-mask').addEventListener('change',e=>setMask(e.target.checked));$('#at-mask2').addEventListener('change',e=>setMask(e.target.checked));
  $('#at-step').addEventListener('click',()=>{attnStop();s4.step=(s4.step+1)%5;renderAttn();});
  $('#at-reset').addEventListener('click',()=>{attnStop();s4.step=0;renderAttn();});
  $('#at-play').addEventListener('click',()=>{if(s4.timer){attnStop();renderAttn();return;}if(s4.step>=4)s4.step=0;s4.timer=setInterval(()=>{if(s4.step>=4){attnStop();}else s4.step++;renderAttn();},1600);renderAttn();});
  $('#at-mat').addEventListener('click',e=>{const c=e.target.closest('.hc');if(c){s4.q=+c.dataset.i;renderAttn();}});
  reg(renderAttn);
}
function initAttnKeys(){
  gridKeys('#at-mat');initAttnLink();
  ['#at-toks','#at-steps'].forEach(sel=>$(sel).addEventListener('click',e=>{if(e.target.closest('button')){const n=$(sel+' button.on');if(n)n.focus();}}));
}
/* ================= multi-head attention ================= */
const s5={h:4};
function renderMH(){
  const heads=MH.heads(s5.h),dh=8/s5.h;
  $('#mh-h').innerHTML=[1,2,4].map(h=>'<button class="chip'+(h===s5.h?' on':'')+'" data-h="'+h+'" aria-pressed="'+(h===s5.h)+'">'+h+'</button>').join('');
  $('#mh-split').innerHTML=heads.map((_,k)=>'<div><div class="cells">'+Array.from({length:dh},(_,c)=>'<i>'+(k*dh+c)+'</i>').join('')+'</div>head '+(k+1)+' · d<sub>k</sub> = '+dh+'</div>').join('');
  $('#mh-grid').innerHTML=heads.map((hd,k)=>'<figure class="mhf"><figcaption><b>Head '+(k+1)+'</b> · '+hd.name+'<small>'+hd.note+'</small></figcaption>'+heat(hd.A,MH.tok,MH.tok.map(t=>t.slice(0,3)),{compact:true,mask:(i,j)=>j>i,tip:(i,j,v,mk)=>'head '+(k+1)+': <b>'+MH.tok[i]+'</b> → <b>'+MH.tok[j]+'</b><br>weight '+v.toFixed(4)+(mk?' (masked)':'')+'<br>score '+hd.S[i][j].toFixed(2)})+'</figure>').join('');
  $('#mh-out').innerHTML='<table class="numt"><thead><tr><th></th>'+heads.map((_,k)=>'<th class="band" colspan="'+dh+'">head '+(k+1)+' output</th>').join('')+'</tr></thead><tbody>'+MH.tok.map((t,i)=>'<tr><th scope="row">'+t+'</th>'+heads.map(hd=>hd.out[i].map((v,c)=>'<td'+(c===0?' class="band"':'')+'>'+fmt(v,2)+'</td>').join('')).join('')+'</tr>').join('')+'</tbody></table>';
}
function initMH(){$('#mh-h').addEventListener('click',e=>{const b=e.target.closest('button');if(b){s5.h=+b.dataset.h;renderMH();$('#mh-h button[data-h="'+s5.h+'"]').focus();}});reg(renderMH);}
/* ================= block: FFN, residuals, normalization ================= */
const s6={x:-1,ln:[0.8,-1.2,2.1,0.3,-0.6,1.5,-2.0,0.4],scale:1,shift:0,res:true,norm:true,gain:1,depth:24,seed:3};
function renderAct(){
  const pts=[],pr=[];for(let x=-4;x<=4.001;x+=0.1){pts.push([x,gelu(x)]);pr.push([x,Math.max(0,x)]);}
  lineChart($('#ac-svg'),[{pts:pr,cls:'b'},{pts:pts,cls:'a'}],{x0:-4,x1:4,y0:-1,y1:4,h:220,xt:[-4,-2,0,2,4],yt:[-1,0,1,2,3,4],xl:'input x',dots:[[s6.x,gelu(s6.x),'GELU']]});
  $('#ac-read').textContent='x = '+fmt(s6.x,1)+'   ReLU(x) = '+fmt(Math.max(0,s6.x))+'   GELU(x) = '+fmt(gelu(s6.x));
}
function renderLN(){
  const x=s6.ln.map(v=>v*s6.scale+s6.shift),n=x.length,mu=x.reduce((a,b)=>a+b,0)/n,sd=Math.sqrt(x.reduce((a,b)=>a+(b-mu)*(b-mu),0)/n+1e-5),y=x.map(v=>(v-mu)/sd);
  signedBars($('#ln-before'),x,Math.max(1,Math.max.apply(null,x.map(Math.abs))));signedBars($('#ln-after'),y,2.6);
  const mu2=y.reduce((a,b)=>a+b,0)/n,sd2=Math.sqrt(y.reduce((a,b)=>a+(b-mu2)*(b-mu2),0)/n);
  $('#ln-read').innerHTML='before: mean μ = '+fmt(mu)+', std σ = '+fmt(sd)+'<br>after:&nbsp; mean = '+fmt(mu2)+', std = '+fmt(sd2);
}
function renderBlock(){
  $('#bl-svg').classList.toggle('no-res',!s6.res);$('#bl-svg').classList.toggle('no-norm',!s6.norm);
  const r=signalSim({L:s6.depth,d:32,gain:s6.gain,residual:s6.res,norm:s6.norm,seed:s6.seed}),L=s6.depth;
  const lg=v=>{const x=Math.log10(v);return isNaN(x)?99:x;};
  const f=r.fwd.map((v,l)=>[l,lg(v/r.fwd[0])]),g=r.grad.map((v,l)=>[l,lg(v/r.grad[L])]);
  let mx=2;f.concat(g).forEach(p=>{mx=Math.max(mx,Math.min(12,Math.abs(p[1])));});mx=Math.ceil(mx/2)*2;
  lineChart($('#bl-chart'),[{pts:f,cls:'a'},{pts:g,cls:'b'}],{x0:0,x1:L,y0:-mx,y1:mx,h:260,ml:50,xt:[0,Math.round(L/2),L],yt:[-mx,-mx/2,0,mx/2,mx],yf:v=>v===0?'×1':'×'+pow10(v),xl:'layer (0 = input, '+L+' = output)'});
  const a=f[L][1],b=g[0][1],worst=Math.abs(a)>Math.abs(b)?a:b,ok=Math.abs(worst)<1.7;
  $('#bl-read').innerHTML='Signal at the output: ×'+sci(Math.pow(10,a))+' the input.<br>Gradient at the first layer: ×'+sci(Math.pow(10,b))+' the output gradient.<br><span class="badge '+(ok?'ok':'bad')+'">'+(ok?'stable':worst>0?(worst>4?'exploding':'growing'):'vanishing')+'</span> '+(ok?'Both stay within a small factor of 1.':worst>0?'Values grow with depth; in a real model they would overflow or swamp the learning rate.':'Values shrink towards zero; early layers would receive almost no learning signal.');
}
function initBlock(){
  slider('ac-x',v=>v.toFixed(1),v=>{s6.x=v;renderAct();});
  slider('ln-scale',v=>'× '+Math.pow(10,v).toFixed(2),v=>{s6.scale=Math.pow(10,v);renderLN();});
  slider('ln-shift',v=>(v>=0?'+':'−')+Math.abs(v).toFixed(1),v=>{s6.shift=v;renderLN();});
  $('#ln-rand').addEventListener('click',()=>{s6.ln=s6.ln.map(()=>Math.round(randn(Math.random)*150)/100);renderLN();});
  $('#bl-res').addEventListener('change',e=>{s6.res=e.target.checked;renderBlock();});
  $('#bl-norm').addEventListener('change',e=>{s6.norm=e.target.checked;renderBlock();});
  slider('bl-gain',v=>v.toFixed(1),v=>{s6.gain=v;renderBlock();});slider('bl-depth',v=>v,v=>{s6.depth=v;renderBlock();});
  $('#bl-seed').addEventListener('click',()=>{s6.seed=Math.floor(Math.random()*1e6);renderBlock();});
  reg(renderAct);reg(renderLN);reg(renderBlock);
}
/* ================= architecture and parameter count ================= */
const R=String.raw;
const archToks=()=>TOK.tokenize('The cat sat on the');
const ARCH=[
  {name:'Text',sub:'a string',what:'The raw input: a string of characters. Nothing is numeric yet.',io:'In: nothing. Out: a string.',par:'None.',tex:null,
   mini:()=>'<div class="sample">The cat sat on the</div>'},
  {name:'Tokenizer',sub:'string → [B, T] integers',what:'Splits the text into tokens from a fixed vocabulary and replaces each with its integer ID. It is a deterministic algorithm, not a neural network.',io:'In: a string. Out: T token IDs.',par:'None that are trained with the model. The vocabulary is fixed before training.',tex:R`w_1,\dots,w_T=\mathrm{tokenize}(\text{text})`,
   mini:()=>'<div class="chips">'+archToks().map(t=>'<span class="chip tok"><span>'+esc(t.t)+'</span><small>id '+t.id+'</small></span>').join('')+'</div>'},
  {name:'Token embeddings',sub:'[B, T] → [B, T, d_model]',what:'Replaces every token ID with its learned vector: one row of the embedding table per token.',io:'In: token IDs. Out: one vector of length d_model per token.',par:'E, an n_vocab × d_model matrix.',tex:R`x_t=E[w_t]`,
   mini:()=>archToks().slice(0,3).map(t=>'<div style="margin:3px 0"><code>'+esc(t.t)+'</code> '+vec(TOK.E[t.id],'',2,1.5)+'</div>').join('')},
  {name:'+ Positional embeddings',sub:'[B, T, d_model] + [T, d_model] → [B, T, d_model]',what:'Adds a vector that encodes each token’s position, so that word order becomes visible to attention.',io:'In: token vectors. Out: token vectors that also carry position.',par:'E_pos, a T × d_model matrix if learned; none if sinusoidal.',tex:R`x^{(0)}_t=E[w_t]+E_{\text{pos}}[t]`,
   mini:()=>{const t=archToks()[1],p=[0,1,2,3].map(i=>pe(1,i,4)),s=TOK.E[t.id].map((v,i)=>v+p[i]);return '<div><code>'+esc(t.t)+'</code> '+vec(TOK.E[t.id],'',2,1.5)+'</div><div>+ slot 1 '+vec(p,'r',2,1.5)+'</div><div>= &nbsp; '+vec(s,'ctx',2,1.5)+'</div>';}},
  {name:'L × Transformer blocks',sub:'[B, T, d_model] → [B, T, d_model]',cls:'blk',what:'The core of the model. Each block lets tokens exchange information through masked multi-head attention, then processes each token with an MLP. Both results are added to the residual stream. The same structure repeats L times with different weights.',io:'In: the residual stream. Out: the residual stream, refined.',par:'Per block: 4(d_model² + d_model) for attention, 2·d_model·d_ff + d_ff + d_model for the MLP, 4·d_model for two LayerNorms.',tex:R`x'=x^{(\ell-1)}+\mathrm{MultiHead}(\mathrm{LayerNorm}(x^{(\ell-1)})),\quad x^{(\ell)}=x'+\mathrm{FFN}(\mathrm{LayerNorm}(x'))`,
   mini:()=>{const keep=s4.mask;s4.mask=true;const A=AT.tok.map((_,i)=>attnCalc(i).w);s4.mask=keep;return '<small>Inside one block: a causal attention pattern (percent)</small>'+heat(A,AT.tok,AT.tok,{compact:true,mask:(i,j)=>j>i});}},
  {name:'Final LayerNorm',sub:'[B, T, d_model] → [B, T, d_model]',what:'Normalises the residual stream one last time, so that the output projection sees vectors of a consistent scale.',io:'In: final residual stream. Out: normalised vectors.',par:'γ and β, 2·d_model numbers.',tex:R`\mathrm{LayerNorm}(x)=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta`,
   mini:()=>{const x=[2,4,6,8],sd=Math.sqrt(5);return '<div>before '+vec(x,'',1,8)+'</div><div>after &nbsp; '+vec(x.map(v=>(v-5)/sd),'ctx',null,2)+'</div>';}},
  {name:'Output projection',sub:'[B, T, d_model] → [B, T, n_vocab]',what:'A linear layer that scores every vocabulary entry against each position’s final vector. The scores are called logits. It is often called the unembedding and frequently shares its weights with the token embedding table.',io:'In: one vector of length d_model per position. Out: n_vocab logits per position.',par:'W_U, a d_model × n_vocab matrix (zero extra if shared with E).',tex:R`z=\mathrm{LayerNorm}(x^{(L)})\,W_U`,
   mini:()=>{const P=NEXT[0],mn=Math.min.apply(null,P.logits),mx=Math.max.apply(null,P.logits);return '<small>Illustrative logits at the last position</small>'+bars(P.toks.map((t,i)=>({label:t,p:P.logits[i]-mn,right:fmt(P.logits[i],1)})),mx-mn);}},
  {name:'Softmax',sub:'[B, T, n_vocab] → [B, T, n_vocab]',what:'Turns each position’s logits into probabilities that are positive and sum to 1.',io:'In: logits. Out: probabilities.',par:'None.',tex:R`p_v=\frac{e^{z_v}}{\sum_{v'} e^{z_{v'}}}`,
   mini:()=>{const P=NEXT[0],p=softmax(P.logits);return '<small>The same logits after softmax</small>'+bars(P.toks.map((t,i)=>({label:t,p:p[i]})),1);}},
  {name:'Next-token distribution',sub:'n_vocab probabilities → 1 token',what:'The probabilities at the last position describe what comes next. A decoding rule picks one token, it is appended to the input, and the whole pipeline runs again.',io:'In: probabilities at the last position. Out: one chosen token.',par:'None. Temperature, top-k and top-p are settings, not learned weights.',tex:R`w_{T+1}\sim p(\,\cdot\mid w_1,\dots,w_T)`,
   mini:()=>{const P=NEXT[0],p=softmax(P.logits),r=mulberry32(4);return '<small>Ten independent samples from that distribution</small><div class="chips">'+Array.from({length:10},()=>'<span class="chip">'+P.toks[sampleFrom(p,r)]+'</span>').join('')+'</div>';}}
];
const s7={sel:4,timer:null};
function renderArch(){
  $('#ar-list').innerHTML=ARCH.map((a,i)=>'<li><button data-i="'+i+'" class="'+(a.cls||'')+(i===s7.sel?' on':'')+(i===s7.pulse?' pulse':'')+'" aria-pressed="'+(i===s7.sel)+'">'+a.name+'<small><code>'+a.sub+'</code></small></button></li>').join('');
  const a=ARCH[s7.sel];
  $('#ar-detail').innerHTML='<h3 style="margin-top:0">'+a.name+'</h3><p>'+a.what+'</p><dl><dt>In / out</dt><dd>'+sym(a.io)+'</dd><dt>Parameters</dt><dd>'+sym(a.par)+'</dd><dt>Shapes</dt><dd><code>'+a.sub+'</code></dd>'+(a.tex?'<dt>Equation</dt><dd class="scroll"><span class="tex">'+esc(a.tex)+'</span></dd>':'')+'</dl><span class="lab">Mini visualisation</span>'+a.mini();
  renderTex($('#ar-detail'));
  $('#ar-play').textContent=s7.timer?'⏸ Pause':'▶ Play data flow';
}
const PRESETS=[['Toy example',{V:50,d:8,L:1,h:2,ff:32,T:16,tie:false}],['GPT-2 small',{V:50257,d:768,L:12,h:12,ff:3072,T:1024,tie:true}],['GPT-2 medium',{V:50257,d:1024,L:24,h:16,ff:4096,T:1024,tie:true}],['GPT-2 large',{V:50257,d:1280,L:36,h:20,ff:5120,T:1024,tie:true}],['GPT-2 XL',{V:50257,d:1600,L:48,h:25,ff:6400,T:1024,tie:true}]];
function renderParams(){
  const num=(id,lo)=>{const v=Math.floor(+$('#pc-'+id).value);return isFinite(v)&&v>=lo?v:lo;};
  const c={V:num('V',2),d:num('d',2),L:num('L',1),h:num('h',1),ff:num('ff',1),T:num('T',1),tie:$('#pc-tie').checked},p=paramCounts(c);
  const rows=[['Token embeddings','n_vocab·d_model',p.tok],['Position embeddings','T·d_model',p.pos],['Attention (all layers)','L·4(d_model²+d_model)',p.attn],['MLP (all layers)','L·(2·d_model·d_ff+d_ff+d_model)',p.mlp],['LayerNorm (all)','L·4·d_model+2·d_model',p.ln],['Output projection',c.tie?'shared with embeddings':'n_vocab·d_model',p.out]];
  $('#pc-warn').innerHTML=sym(c.d%c.h?'<span class="badge bad">check</span> d_model must be divisible by the number of heads: '+c.d+' / '+c.h+' is not a whole number.':'Each head works with d_model / h = '+(c.d/c.h)+' dimensions. The number of heads does not appear in any formula below.');
  $('#pc-bar').innerHTML=rows.map((r,i)=>{const w=100*r[2]/p.total;return w>0?'<span class="p'+i+'" style="width:'+w+'%" data-tip="'+esc(r[0]+': '+r[2].toLocaleString('en-US')+' ('+w.toFixed(1)+'%)')+'" tabindex="0">'+(w>9?w.toFixed(0)+'%':'')+'</span>':'';}).join('');
  $('#pc-table').innerHTML='<thead><tr><th>Component</th><th>Formula</th><th>Parameters</th><th>Share</th></tr></thead><tbody>'+rows.map((r,i)=>'<tr><th scope="row"><i class="psw p'+i+'"></i>'+r[0]+'</th><td>'+sym(r[1])+'</td><td>'+r[2].toLocaleString('en-US')+'</td><td>'+(100*r[2]/p.total).toFixed(1)+'%</td></tr>').join('')+'<tr><th scope="row">Total</th><td></td><td><b>'+p.total.toLocaleString('en-US')+'</b></td><td>≈ '+big(p.total)+'</td></tr></tbody>';
  $('#pc-mem').innerHTML=[['32-bit floats',4],['16-bit floats',2],['8-bit quantized',1],['4-bit quantized',0.5]].map(m=>'<div><span>'+m[0]+'</span><b>'+bytes(p.total*m[1])+'</b></div>').join('');
  $('#pc-scale').innerHTML='<thead><tr><th>'+sym('d_model (d_ff scaled with it)')+'</th><th>Total parameters</th><th>Relative</th></tr></thead><tbody>'+[0.5,1,2,4].map(k=>{const q=paramCounts(Object.assign({},c,{d:Math.round(c.d*k),ff:Math.round(c.ff*k)})).total;return '<tr><th scope="row">'+Math.round(c.d*k)+(k===1?' (current)':'')+'</th><td>'+big(q)+'</td><td>× '+(q/p.total).toFixed(2)+'</td></tr>';}).join('')+'</tbody>';
}
function initArch(){
  $('#ar-list').addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;if(s7.timer){clearInterval(s7.timer);s7.timer=null;}s7.pulse=-1;s7.sel=+b.dataset.i;renderArch();$('#ar-list button[data-i="'+s7.sel+'"]').focus();});
  $('#ar-play').addEventListener('click',()=>{if(s7.timer){clearInterval(s7.timer);s7.timer=null;s7.pulse=-1;renderArch();return;}s7.sel=0;s7.pulse=0;s7.timer=setInterval(()=>{if(s7.sel>=ARCH.length-1){clearInterval(s7.timer);s7.timer=null;s7.pulse=-1;}else{s7.sel++;s7.pulse=s7.sel;}renderArch();},1500);renderArch();});
  $('#pc-presets').innerHTML=PRESETS.map((p,i)=>'<button data-i="'+i+'">'+p[0]+'</button>').join('');
  $('#pc-presets').addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;const c=PRESETS[+b.dataset.i][1];['V','d','L','h','ff','T'].forEach(k=>{$('#pc-'+k).value=c[k];});$('#pc-tie').checked=c.tie;renderParams();});
  $$('#w-params input').forEach(i=>i.addEventListener('input',renderParams));
  reg(renderArch);reg(renderParams);
}

/* ================= BPE: learn a vocabulary, merge by merge ================= */
const BPE_PRESETS={
  classic:'low low low low low lower lower newest newest newest newest newest newest widest widest widest',
  text:'the cat sat on the mat. the cats sat on the mats. the dog sat on the log. a cat and a dog sat on a mat. the dogs and the cats sat.'};
const sbp={preset:'classic',merges:[],word:'lowest'};
function bpeWords(text){const c={};(text.toLowerCase().match(/[a-z]+/g)||[]).forEach(w=>{c[w]=(c[w]||0)+1;});return Object.keys(c).map(w=>({w,n:c[w]}));}
function bpeMerge(p,a,b){const o=[];for(let i=0;i<p.length;i++){if(i<p.length-1&&p[i]===a&&p[i+1]===b){o.push(a+b);i++;}else o.push(p[i]);}return o;}
function bpeState(){
  const words=bpeWords($('#bp-text').value).map(o=>({w:o.w,n:o.n,p:o.w.split('')}));
  sbp.merges.forEach(([a,b])=>words.forEach(o=>{o.p=bpeMerge(o.p,a,b);}));
  const pairs=new Map();
  words.forEach(o=>{for(let i=0;i<o.p.length-1;i++){const k=o.p[i]+'\u0001'+o.p[i+1];pairs.set(k,(pairs.get(k)||0)+o.n);}});
  const list=[...pairs].map(([k,n])=>({a:k.split('\u0001')[0],b:k.split('\u0001')[1],n}));
  const best=list.reduce((m,x)=>!m||x.n>m.n?x:m,null);             // ties go to the pair seen first
  return{words,list:list.slice().sort((x,y)=>y.n-x.n),best};
}
function bpeEncode(word){let p=word.toLowerCase().replace(/[^a-z]/g,'').split('');const used=[];sbp.merges.forEach(([a,b],i)=>{const q=bpeMerge(p,a,b);if(q.length<p.length)used.push(i+1);p=q;});return{p,used};}
function renderBpe(){
  const S=bpeState(),chars=new Set();S.words.forEach(o=>o.w.split('').forEach(c=>chars.add(c)));
  const piece=t=>'<span class="bpp">'+esc(t)+'</span>';
  $('#bp-pairs').innerHTML=S.list.length?bars(S.list.slice(0,6).map(x=>({label:esc(x.a)+' + '+esc(x.b),p:x.n,cls:x===S.best?'hit':'',right:x.n+'×'+(x===S.best?' · next merge':'')})),S.best.n):'<p class="cap">No pairs left: every word is a single token.</p>';
  const ws=S.words.slice().sort((x,y)=>y.n-x.n).slice(0,12);
  $('#bp-words').innerHTML='<table class="numt"><thead><tr><th>Word</th><th>Count</th><th>Current tokens</th></tr></thead><tbody>'+ws.map(o=>'<tr><td>'+esc(o.w)+'</td><td>'+o.n+'</td><td>'+o.p.map(piece).join('')+'</td></tr>').join('')+'</tbody></table>'+(S.words.length>12?'<small>The 12 most frequent of '+S.words.length+' different words.</small>':'');
  $('#bp-merges').innerHTML=sbp.merges.length?sbp.merges.map(([a,b],i)=>'<span class="chip tok">'+esc(a)+' + '+esc(b)+' → '+esc(a+b)+'<small>merge '+(i+1)+'</small></span>').join(''):'<small>None yet. Press “Merge the top pair”.</small>';
  const total=S.words.reduce((s,o)=>s+o.n*o.p.length,0),nw=S.words.reduce((s,o)=>s+o.n,0);
  $('#bp-stats').innerHTML=[['Vocabulary: letters + merges',chars.size+' + '+sbp.merges.length+' = '+(chars.size+sbp.merges.length)],['Tokens for the whole text',total],['Tokens per word',nw?fmt(total/nw,2):'–']].map(r=>'<div><b>'+r[1]+'</b><span>'+r[0]+'</span></div>').join('');
  const E=bpeEncode(sbp.word);
  $('#bp-enc').innerHTML=E.p.length?E.p.map(piece).join('')+' <small>'+E.p.length+' token'+(E.p.length>1?'s':'')+(E.used.length?'; merges used, in order: '+E.used.join(', '):'; no learned merge applies')+'</small>':'<small>Type a word made of letters.</small>';
  $('#bp-next').disabled=$('#bp-ten').disabled=!S.best;$('#bp-undo').disabled=!sbp.merges.length;
  $$('#bp-preset button').forEach(b=>{const on=b.dataset.v===sbp.preset;b.classList.toggle('on',on);b.setAttribute('aria-pressed',on);});
}
function initBpe(){
  const el=$('#bp-preset');
  el.innerHTML=[['classic','The example above'],['text','Short sentences']].map(o=>'<button class="chip" data-v="'+o[0]+'" aria-pressed="false">'+o[1]+'</button>').join('');
  el.addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;sbp.preset=b.dataset.v;$('#bp-text').value=BPE_PRESETS[sbp.preset];sbp.merges=[];sbp.word=$('#bp-word').value=sbp.preset==='classic'?'lowest':'hats';renderBpe();$('#bp-preset button[data-v="'+sbp.preset+'"]').focus();});
  $('#bp-text').value=BPE_PRESETS.classic;
  $('#bp-text').addEventListener('input',()=>{sbp.preset='';sbp.merges=[];renderBpe();});
  const step=()=>{const S=bpeState();if(S.best)sbp.merges.push([S.best.a,S.best.b]);};
  $('#bp-next').addEventListener('click',()=>{step();renderBpe();});
  $('#bp-ten').addEventListener('click',()=>{for(let i=0;i<10;i++)step();renderBpe();});
  $('#bp-undo').addEventListener('click',()=>{sbp.merges.pop();renderBpe();});
  $('#bp-reset').addEventListener('click',()=>{sbp.merges=[];renderBpe();});
  $('#bp-word').addEventListener('input',e=>{sbp.word=e.target.value;renderBpe();});
  reg(renderBpe);
}

PAGE_INITS.push(initTok,initBpe,initPE,initAttn,initAttnKeys,initMH,initBlock,initArch);
