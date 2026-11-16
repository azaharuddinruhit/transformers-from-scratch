'use strict';
/* Part D widgets: encoder–decoder cross-attention and the 2017 paper's learning-rate schedule. */
/* ================= encoder-decoder and cross-attention ================= */
const XA={src:['le','chat','noir','dort'],din:['<s>','the','black','cat'],out:['the','black','cat','sleeps'],
  Q:[[2,1,0,0],[0,0,2,0],[0,2,0,0],[0,1,0,2]],K:[[2,0,0,0],[0,2,1,0],[0,1,2,0],[0,0,0,2]],V:[[1,0,0,0.5],[0,2,0,1],[0,0,2,-1],[0.5,1,0,2]]};
const sx={q:1,view:'ed'};
function xCalc(i){const dots=XA.K.map(k=>dot(XA.Q[i],k)),scaled=dots.map(s=>s/2),exps=scaled.map(Math.exp),w=softmax(scaled);return {dots:dots,scaled:scaled,exps:exps,w:w,ctx:[0,1,2,3].map(c=>w.reduce((s,a,j)=>s+a*XA.V[j][c],0))};}
function renderCross(){
  const i=sx.q,c=xCalc(i),X=state.exact,best=c.w.indexOf(Math.max.apply(null,c.w));
  $('#xa-toks').innerHTML=XA.out.map((t,j)=>'<button class="chip'+(j===i?' on':'')+'" data-i="'+j+'" aria-pressed="'+(j===i)+'">predict “'+t+'”</button>').join('');
  $('#xa-cap').innerHTML='The decoder has read “'+XA.din.slice(0,i+1).map(esc).join(' ')+'” and is predicting <b>“'+XA.out[i]+'”</b>. Its <b class="tq">query</b> is compared with the <b class="tk">keys</b> of all four source tokens; the strongest match is <b>“'+XA.src[best]+'”</b> at '+Math.round(c.w[best]*100)+'%.';
  const svg=$('#xa-svg'),W=Math.max(250,Math.round(svg.getBoundingClientRect().width)||600),H=214,bw=Math.min(110,W/4-10),cx=k=>(k+0.5)*W/4;
  svg.setAttribute('viewBox','0 0 '+W+' '+H);svg.style.height=H+'px';
  let s='<text class="ax" x="'+W/2+'" y="12" text-anchor="middle">encoder output: keys and values (source sentence)</text><text class="ax" x="'+W/2+'" y="'+(H-3)+'" text-anchor="middle">decoder: query (target so far → next word)</text>';
  XA.src.forEach((t,j)=>{const w=c.w[j];s+='<line x1="'+cx(i)+'" y1="150" x2="'+cx(j)+'" y2="56" stroke="var(--accent)" stroke-linecap="round" stroke-width="'+(1+13*w).toFixed(1)+'" opacity="'+(0.25+0.75*w).toFixed(2)+'"/>';});
  XA.src.forEach((t,j)=>{s+='<g class="sr"><rect x="'+(cx(j)-bw/2)+'" y="20" width="'+bw+'" height="36" rx="8"/><text x="'+cx(j)+'" y="43" text-anchor="middle">'+t+'</text></g><text class="pc" x="'+cx(j)+'" y="74" text-anchor="middle">'+Math.round(c.w[j]*100)+'%</text>';});
  XA.out.forEach((t,j)=>{s+='<g class="tg'+(j===i?' on':'')+'" tabindex="0" role="button" aria-label="Decoder step '+(j+1)+': predict '+t+'" data-i="'+j+'"><rect x="'+(cx(j)-bw/2)+'" y="150" width="'+bw+'" height="44" rx="8"/><text x="'+cx(j)+'" y="168" text-anchor="middle">'+esc(XA.din[j])+'</text><text class="ax" x="'+cx(j)+'" y="186" text-anchor="middle">→ '+t+'</text></g>';});
  svg.innerHTML=s;
  $('#xa-q').innerHTML='<span class="lab">Decoder query at this step</span><b class="tq">q</b> = '+vec(XA.Q[i],'q',1,2);
  let h='<thead><tr><th>Source token j</th><th><span class="tk">Key k<sub>j</sub></span></th><th>q · k<sub>j</sub></th><th>÷ 2.00</th>'+(X?'<th>exp</th>':'')+'<th>Weight α<sub>j</sub></th><th><span class="tv">Value v<sub>j</sub></span></th></tr></thead><tbody>';
  XA.src.forEach((t,j)=>{const ex=XA.Q[i].map((a,x)=>a+'·'+XA.K[j][x]).join(' + ');
    h+='<tr><th scope="row">'+t+'</th><td>'+vec(XA.K[j],'k',1,2)+'</td><td><b>'+fmt(c.dots[j],0)+'</b>'+(X?'<small class="ex">'+ex+'</small>':'')+'</td><td>'+fmt(c.scaled[j])+'</td>'+(X?'<td>'+fmt(c.exps[j])+'</td>':'')+'<td'+(j===best?' class="cur"':'')+'><b>'+fmt(c.w[j])+'</b></td><td>'+vec(XA.V[j],'v',1,2)+'</td></tr>';});
  if(X)h+='<tr><th scope="row">Σ</th><td></td><td></td><td></td><td>'+fmt(c.exps.reduce((a,b)=>a+b,0))+'</td><td>'+fmt(c.w.reduce((a,b)=>a+b,0))+'</td><td></td></tr>';
  $('#xa-table').innerHTML=h+'</tbody>';
  $('#xa-soft').innerHTML=bars(XA.src.map((t,j)=>({label:t,p:c.w[j],cls:j===best?'hit':'',tip:'α = '+c.w[j].toFixed(6)})),1);
  $('#xa-mat').innerHTML=heat(XA.out.map((_,k)=>xCalc(k).w),XA.out.map(t=>'→ '+t),XA.src,{click:true,sel:i,tip:(a,b,v)=>'decoder step <b>→ '+XA.out[a]+'</b> attends to source <b>'+XA.src[b]+'</b><br>weight '+v.toFixed(6)});
  $('#xa-ctx').innerHTML='<span class="lab">Context vector handed to the decoder: c = Σ α<sub>j</sub> v<sub>j</sub></span>'+vec(c.ctx,'ctx',null,2)+(X?'<div class="readout">'+[0,1,2,3].map(d=>'c<sub>'+d+'</sub> = '+XA.src.map((t,j)=>fmt(c.w[j])+'·'+String(XA.V[j][d]).replace('-','−')).join(' + ')+' = <b>'+fmt(c.ctx[d])+'</b>').join('<br>')+'</div>':'<p><small>A blend of the encoder’s value vectors, dominated by “'+XA.src[best]+'”. It is added to this decoder position’s residual stream.</small></p>');
}
function typeGrid(rows,cols,fn){
  return '<div class="hm maskg'+(cols.length>5?' compact':'')+'" style="grid-template-columns:auto repeat('+cols.length+',minmax(0,1fr))"><span></span>'+cols.map(t=>'<span class="hh">'+esc(t.length>4&&cols.length>5?t.slice(0,3):t)+'</span>').join('')+rows.map((t,i)=>'<span class="hr">'+esc(t)+'</span>'+cols.map((u,j)=>{const k=fn(i,j);return '<span class="hc'+(k?'':' masked')+'" data-tip="'+esc('<b>'+esc(t)+'</b> → <b>'+esc(u)+'</b><br>'+({S:'source reads source',X:'target reads source',T:'target reads earlier target'}[k]||'blocked: −∞ added before softmax'))+'">'+(k||'−∞')+'</span>';}).join('')).join('')+'</div>';
}
function renderContrast(){
  const ed=sx.view==='ed';
  $('#xc-view').innerHTML=[['ed','Encoder–decoder'],['do','Decoder-only']].map(v=>'<button class="chip'+(v[0]===sx.view?' on':'')+'" data-v="'+v[0]+'" aria-pressed="'+(v[0]===sx.view)+'">'+v[1]+'</button>').join('');
  if(ed){
    $('#xc-out').innerHTML='<div class="grid2"><div><span class="lab">Encoder self-attention (no mask)</span>'+typeGrid(XA.src,XA.src,()=>'S')+'</div><div><span class="lab">Cross-attention (no mask)</span>'+typeGrid(XA.din,XA.src,()=>'X')+'</div><div><span class="lab">Decoder self-attention (causal mask)</span>'+typeGrid(XA.din,XA.din,(i,j)=>j>i?'':'T')+'</div></div>';
    $('#xc-read').innerHTML='Three tables, three sets of attention weights. Allowed connections: 16 source→source + 16 target→source + 10 target→target = <b>42</b>. Every source token sees the whole source sentence.';
  }else{
    const seq=XA.src.concat(XA.din);
    $('#xc-out').innerHTML='<span class="lab">One causal self-attention table over the joined sequence</span><div style="max-width:560px">'+typeGrid(seq,seq,(i,j)=>j>i?'':i<4?'S':j<4?'X':'T')+'</div>';
    $('#xc-read').innerHTML='One table, one set of attention weights. Allowed connections: 10 source→source + 16 target→source + 10 target→target = <b>36</b>. The 6 that disappeared are source tokens looking at later source tokens: “le” can no longer see “chat”.';
  }
}
function initCross(){
  $('#xa-toks').addEventListener('click',e=>{const b=e.target.closest('button');if(b){sx.q=+b.dataset.i;renderCross();$('#xa-toks button[data-i="'+sx.q+'"]').focus();}});
  const pick=e=>{const g=e.target.closest('.tg');if(!g)return;if(e.type==='keydown'){if(e.key!=='Enter'&&e.key!==' ')return;e.preventDefault();}sx.q=+g.dataset.i;renderCross();if(e.type==='keydown')$('#xa-svg .tg[data-i="'+sx.q+'"]').focus();};
  $('#xa-svg').addEventListener('click',pick);$('#xa-svg').addEventListener('keydown',pick);
  $('#xa-mat').addEventListener('click',e=>{const c=e.target.closest('.hc');if(c){sx.q=+c.dataset.i;renderCross();}});
  $('#xc-view').addEventListener('click',e=>{const b=e.target.closest('button');if(b){sx.view=b.dataset.v;renderContrast();$('#xc-view button[data-v="'+sx.view+'"]').focus();}});
  reg(renderCross);reg(renderContrast);
}
function initCrossKeys(){gridKeys('#xa-mat');}
/* ================= the 2017 paper ================= */
const sp={d:512,w:4000,s:4000,};
const lrate=(d,w,s)=>Math.pow(d,-0.5)*Math.min(Math.pow(s,-0.5),s*Math.pow(w,-1.5));
const sci3=x=>{const e=Math.floor(Math.log10(x));return (x/Math.pow(10,e)).toFixed(3)+'×'+pow10(e);};
function renderLR(){
  const S=100000,xs=[sp.w];for(let s=1;s<=S;s+=s<2000?50:500)xs.push(s);xs.push(S);xs.sort((a,b)=>a-b);
  const peak=lrate(sp.d,sp.w,sp.w),top=peak*1.15,cur=lrate(sp.d,sp.w,sp.s),e4=v=>(v*1e4).toFixed(v*1e4<10?2:1);
  lineChart($('#lr-chart'),[{pts:xs.map(s=>[s,lrate(sp.d,sp.w,s)]),cls:'a'}],{x0:0,x1:S,y0:0,y1:top,h:250,ml:52,xt:[0,25000,50000,75000,100000],xf:v=>v?v/1000+'k':'0',yt:[0,top/3,2*top/3,top],yf:e4,xl:'training step',yl:'learning rate (×10⁻⁴)',dots:[[sp.s,cur,'step '+sp.s.toLocaleString('en-US')]]});
  const a=Math.pow(sp.s,-0.5),b=sp.s*Math.pow(sp.w,-1.5);
  $('#lr-read').innerHTML='Step '+sp.s.toLocaleString('en-US')+': learning rate = <b>'+sci3(cur)+'</b> ('+(sp.s<sp.w?'warm-up, rising linearly':sp.s===sp.w?'the peak':'decay, falling as step⁻⁰·⁵')+')'+(state.exact?'<br>'+sp.d+'⁻⁰·⁵ × min('+sci3(a)+', '+sci3(b)+') = '+fmt(Math.pow(sp.d,-0.5),6)+' × '+sci3(Math.min(a,b)):'')+'<br>Peak at step '+sp.w.toLocaleString('en-US')+': '+sci3(peak)+' = ('+sp.d+' × '+sp.w+')⁻⁰·⁵';
}
function initLR(){
  slider('lr-d',v=>Math.pow(2,v),v=>{sp.d=Math.pow(2,v);renderLR();});
  slider('lr-w',v=>v.toLocaleString('en-US'),v=>{sp.w=v;renderLR();});
  slider('lr-s',v=>(v?v*1000:1).toLocaleString('en-US'),v=>{sp.s=v?v*1000:1;renderLR();});
  reg(renderLR);
}

/* ================= Modern LLMs: RoPE rotation ================= */
const RP_PAIRS=[['fast pair · ω = 1',1],['middle pair · ω = 0.1',0.1],['slow pair · ω = 0.01',0.01]],RP_Q=[0.9,0.35],RP_K=[0.75,-0.45];
const srp={i:5,j:1,f:0};
function rot(v,a){return [v[0]*Math.cos(a)-v[1]*Math.sin(a),v[0]*Math.sin(a)+v[1]*Math.cos(a)];}
function renderRope(){
  const w=RP_PAIRS[srp.f][1],q=rot(RP_Q,srp.i*w),k=rot(RP_K,srp.j*w),score=q[0]*k[0]+q[1]*k[1],c=120,R=100;
  const arrow=(v,cls,lab)=>{const x=c+R*v[0],y=c-R*v[1];return '<line x1="'+c+'" y1="'+c+'" x2="'+x.toFixed(1)+'" y2="'+y.toFixed(1)+'" class="rpv '+cls+'"/><circle cx="'+x.toFixed(1)+'" cy="'+y.toFixed(1)+'" r="4" class="rpd '+cls+'"/><text x="'+(c+(R+14)*v[0]).toFixed(1)+'" y="'+(c-(R+14)*v[1]+4).toFixed(1)+'" class="rpt '+cls+'" text-anchor="middle">'+lab+'</text>';};
  $('#rp-svg').innerHTML='<circle cx="'+c+'" cy="'+c+'" r="'+R+'" class="rpc"/><line x1="'+(c-R)+'" y1="'+c+'" x2="'+(c+R)+'" y2="'+c+'" class="rpa"/><line x1="'+c+'" y1="'+(c-R)+'" x2="'+c+'" y2="'+(c+R)+'" class="rpa"/>'+arrow(q,'q','q')+arrow(k,'k','k');
  $('#rp-f').innerHTML=RP_PAIRS.map((p,i)=>'<button class="chip'+(i===srp.f?' on':'')+'" data-f="'+i+'" aria-pressed="'+(i===srp.f)+'">'+p[0]+'</button>').join('');
  const deg=x=>(((x*180/Math.PI)%360)+360)%360;
  $('#rp-stats').innerHTML='<div><b>'+(srp.i-srp.j)+'</b><span>distance i − j</span></div><div><b>'+deg(srp.i*w).toFixed(0)+'° / '+deg(srp.j*w).toFixed(0)+'°</b><span>query / key turned by</span></div><div><b>'+fmt(score,3)+'</b><span>score for this pair</span></div>';
}
function initRope(){
  const set=(id,v)=>{const e=$('#'+id);e.value=v;e.dispatchEvent(new Event('input'));};
  slider('rp-i',v=>String(v),v=>{srp.i=v;renderRope();});slider('rp-j',v=>String(v),v=>{srp.j=v;renderRope();});
  $('#rp-f').addEventListener('click',e=>{const b=e.target.closest('button');if(b){srp.f=+b.dataset.f;renderRope();$('#rp-f button[data-f="'+srp.f+'"]').focus();}});
  $('#rp-shift').addEventListener('click',()=>{if(srp.i>55||srp.j>55){set('rp-i',srp.i-srp.j>=0?srp.i-srp.j:0);set('rp-j',srp.i-srp.j>=0?0:srp.j-srp.i);}else{set('rp-i',srp.i+5);set('rp-j',srp.j+5);}});
  reg(renderRope);
}
/* ================= Modern LLMs: grouped-query attention and the KV cache ================= */
const sgq={kv:8,T:8192,B:1},GQ_W=8.03e9*2;
function renderGqa(){
  const per=2*32*sgq.kv*128*2,kv=per*sgq.T*sgq.B,mha=2*32*32*128*2*sgq.T*sgq.B,span=GQ_W+kv;
  $('#gq-kv').innerHTML=[32,16,8,4,2,1].map(n=>'<button class="chip'+(n===sgq.kv?' on':'')+'" data-n="'+n+'" aria-pressed="'+(n===sgq.kv)+'">'+n+(n===32?' (multi-head)':n===1?' (multi-query)':'')+'</button>').join('');
  const g=32/sgq.kv;let h='';for(let i=0;i<32;i++){const grp=Math.floor(i/g);h+='<span style="background:'+viridis(sgq.kv>1?grp/(sgq.kv-1):0.5)+'" tabindex="0" data-tip="Query head '+(i+1)+' reads key and value head '+(grp+1)+'"></span>';}
  $('#gq-heads').innerHTML=h;
  const seg=(cls,v,label)=>'<span class="'+cls+'" style="width:'+100*v/span+'%" tabindex="0" data-tip="'+esc(label+': '+bytes(v))+'">'+(v/span>0.14?label:'')+'</span>';
  $('#gq-bar').innerHTML=seg('p0',GQ_W,'weights')+seg('p2',kv,'KV cache');
  $('#gq-stats').innerHTML='<div><b>'+(per/1024).toFixed(0)+' KiB</b><span>cache per token</span></div><div><b>'+bytes(kv)+'</b><span>cache in total</span></div><div><b>'+bytes(GQ_W+kv)+'</b><span>weights + cache</span></div><div><b>'+(mha/kv).toFixed(0)+'×</b><span>smaller than multi-head</span></div>';
}
function initGqa(){
  slider('gq-T',v=>Math.pow(2,v).toLocaleString('en-US')+' tokens',v=>{sgq.T=Math.pow(2,v);renderGqa();});
  slider('gq-B',v=>String(Math.pow(2,v)),v=>{sgq.B=Math.pow(2,v);renderGqa();});
  $('#gq-kv').addEventListener('click',e=>{const b=e.target.closest('button');if(b){sgq.kv=+b.dataset.n;renderGqa();$('#gq-kv button[data-n="'+sgq.kv+'"]').focus();}});
  reg(renderGqa);
}

PAGE_INITS.push(initCross,initCrossKeys,initLR,initRope,initGqa);
