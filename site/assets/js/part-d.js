'use strict';
/* Part D widgets: encoder–decoder cross-attention and the 2017 paper's learning-rate schedule. */
/* ================= encoder-decoder and cross-attention ================= */
const XA={src:['le','chat','noir','dort'],din:['<s>','the','black','cat'],out:['the','black','cat','sleeps'],
  Q:[[2,1,0,0],[0,0,2,0],[0,2,0,0],[0,1,0,2]],K:[[2,0,0,0],[0,2,1,0],[0,1,2,0],[0,0,0,2]],V:[[1,0,0,0.5],[0,2,0,1],[0,0,2,-1],[0.5,1,0,2]]};
const sx={q:1,view:'ed'};
function xCalc(i){const dots=XA.K.map(k=>dot(XA.Q[i],k)),scaled=dots.map(s=>s/2),exps=scaled.map(Math.exp),w=softmax(scaled);return {dots:dots,scaled:scaled,exps:exps,w:w,ctx:[0,1,2,3].map(c=>w.reduce((s,a,j)=>s+a*XA.V[j][c],0))};}
function renderCross(){
  const i=sx.q,c=xCalc(i),best=c.w.indexOf(Math.max.apply(null,c.w));
  $('#xa-toks').innerHTML=XA.out.map((t,j)=>'<button class="chip'+(j===i?' on':'')+'" data-i="'+j+'" aria-pressed="'+(j===i)+'">predict “'+t+'”</button>').join('');
  $('#xa-cap').innerHTML='The decoder has read “'+XA.din.slice(0,i+1).map(esc).join(' ')+'” and is predicting <b>“'+XA.out[i]+'”</b>. Its <b class="tq">query</b> is compared with the <b class="tk">keys</b> of all four source tokens; the strongest match is <b>“'+XA.src[best]+'”</b> at '+Math.round(c.w[best]*100)+'%.';
  const svg=$('#xa-svg'),W=Math.max(250,Math.round(svg.getBoundingClientRect().width)||600),H=214,bw=Math.min(110,W/4-10),cx=k=>(k+0.5)*W/4;
  svg.setAttribute('viewBox','0 0 '+W+' '+H);svg.style.height=H+'px';
  let s='<text class="ax" x="'+W/2+'" y="12" text-anchor="middle">encoder output: keys and values (source sentence)</text><text class="ax" x="'+W/2+'" y="'+(H-3)+'" text-anchor="middle">decoder: query (target so far → next word)</text>';
  XA.src.forEach((t,j)=>{const w=c.w[j];s+='<line x1="'+cx(i)+'" y1="150" x2="'+cx(j)+'" y2="56" stroke="'+viridis(.25+.7*w)+'" stroke-linecap="round" stroke-width="'+(1+13*w).toFixed(1)+'" opacity="'+(0.25+0.75*w).toFixed(2)+'"/>';});
  XA.src.forEach((t,j)=>{s+='<g class="sr"><rect x="'+(cx(j)-bw/2)+'" y="20" width="'+bw+'" height="36" rx="8"/><text x="'+cx(j)+'" y="43" text-anchor="middle">'+t+'</text></g><text class="pc" x="'+cx(j)+'" y="74" text-anchor="middle">'+Math.round(c.w[j]*100)+'%</text>';});
  XA.out.forEach((t,j)=>{s+='<g class="tg'+(j===i?' on':'')+'" tabindex="0" role="button" aria-label="Decoder step '+(j+1)+': predict '+t+'" data-i="'+j+'"><rect x="'+(cx(j)-bw/2)+'" y="150" width="'+bw+'" height="44" rx="8"/><text x="'+cx(j)+'" y="168" text-anchor="middle">'+esc(XA.din[j])+'</text><text class="ax" x="'+cx(j)+'" y="186" text-anchor="middle">→ '+t+'</text></g>';});
  svg.innerHTML=s;
  $('#xa-q').innerHTML='<span class="lab">Decoder query at this step</span><b class="tq">q</b> = '+vec(XA.Q[i],'q',1,2);
  let h='<thead><tr><th>Source token j</th><th><span class="tk">Key k<sub>j</sub></span></th><th>q · k<sub>j</sub></th><th>÷ 2.00</th>'+'<th>exp</th>'+'<th>Weight α<sub>j</sub></th><th><span class="tv">Value v<sub>j</sub></span></th></tr></thead><tbody>';
  XA.src.forEach((t,j)=>{const ex=XA.Q[i].map((a,x)=>a+'·'+XA.K[j][x]).join(' + ');
    h+='<tr><th scope="row">'+t+'</th><td>'+vec(XA.K[j],'k',1,2)+'</td><td><b>'+fmt(c.dots[j],0)+'</b>'+'<small class="ex">'+ex+'</small>'+'</td><td>'+fmt(c.scaled[j])+'</td>'+'<td>'+fmt(c.exps[j])+'</td>'+'<td'+(j===best?' class="cur"':'')+'><b>'+fmt(c.w[j])+'</b></td><td>'+vec(XA.V[j],'v',1,2)+'</td></tr>';});
  h+='<tr><th scope="row">Σ</th><td></td><td></td><td></td><td>'+fmt(c.exps.reduce((a,b)=>a+b,0))+'</td><td>'+fmt(c.w.reduce((a,b)=>a+b,0))+'</td><td></td></tr>';
  $('#xa-table').innerHTML=h+'</tbody>';
  $('#xa-soft').innerHTML=bars(XA.src.map((t,j)=>({label:t,p:c.w[j],cls:j===best?'hit':'',tip:'α = '+c.w[j].toFixed(6)})),1);
  $('#xa-mat').innerHTML=heat(XA.out.map((_,k)=>xCalc(k).w),XA.out.map(t=>'→ '+t),XA.src,{click:true,sel:i,tip:(a,b,v)=>'decoder step <b>→ '+XA.out[a]+'</b> attends to source <b>'+XA.src[b]+'</b><br>weight '+v.toFixed(6)});
  $('#xa-ctx').innerHTML='<span class="lab">Context vector handed to the decoder: c = Σ α<sub>j</sub> v<sub>j</sub></span>'+vec(c.ctx,'ctx',null,2)+'<div class="readout">'+[0,1,2,3].map(d=>'c<sub>'+d+'</sub> = '+XA.src.map((t,j)=>fmt(c.w[j])+'·'+String(XA.V[j][d]).replace('-','−')).join(' + ')+' = <b>'+fmt(c.ctx[d])+'</b>').join('<br>')+'</div>';
}
function typeGrid(rows,cols,fn){
  return '<div class="hm maskg'+(cols.length>5?' compact':'')+'" style="grid-template-columns:auto repeat('+cols.length+',minmax(0,1fr))"><span></span>'+cols.map(t=>'<span class="hh">'+esc(t.length>4&&cols.length>5?t.slice(0,3):t)+'</span>').join('')+rows.map((t,i)=>'<span class="hr">'+esc(t)+'</span>'+cols.map((u,j)=>{const k=fn(i,j);return '<span class="hc'+(k?' t'+k:' masked')+'" data-tip="'+esc('<b>'+esc(t)+'</b> → <b>'+esc(u)+'</b><br>'+({S:'source reads source',X:'target reads source',T:'target reads earlier target'}[k]||'blocked: −∞ added before softmax'))+'">'+(k||'−∞')+'</span>';}).join('')).join('')+'</div>';
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
  $('#lr-read').innerHTML='Step '+sp.s.toLocaleString('en-US')+': learning rate = <b>'+sci3(cur)+'</b> ('+(sp.s<sp.w?'warm-up, rising linearly':sp.s===sp.w?'the peak':'decay, falling as step⁻⁰·⁵')+')'+'<br>'+sp.d+'⁻⁰·⁵ × min('+sci3(a)+', '+sci3(b)+') = '+fmt(Math.pow(sp.d,-0.5),6)+' × '+sci3(Math.min(a,b))+'<br>Peak at step '+sp.w.toLocaleString('en-US')+': '+sci3(peak)+' = ('+sp.d+' × '+sp.w+')⁻⁰·⁵';
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
/* ================= Multimodal LLMs: an image becomes tokens ================= */
/* A 48 x 48 picture drawn by code (a cat on a mat by a window), cut into square patches. */
const PT_N=48,PT_Q=['What',' is',' the',' cat',' sitting',' on','?'];
const PT_PIX=(()=>{
  const px=[],inE=(x,y,cx,cy,rx,ry)=>((x-cx)/rx)**2+((y-cy)/ry)**2<=1;
  for(let y=0;y<PT_N;y++){const row=[];for(let x=0;x<PT_N;x++){
    let c=y<32?[232,222,200]:[176,138,98];                                   // wall, floor
    if(y>=32&&(x+3*y)%12===0)c=[156,120,84];                                  // floorboards
    if(x>=3&&x<=17&&y>=3&&y<=17)c=(x===3||x===17||y===3||y===17||x===10||y===10)?[110,82,58]:[158,204,236];
    if(inE(x,y,7,7,2.2,2.2))c=[250,214,84];                                  // sun in the window
    if(x>=6&&x<=43&&y>=37&&y<=44)c=y===40||y===41?[232,188,92]:[186,58,58];  // mat with a stripe
    const tail=x>=8&&x<=14&&Math.abs(y-(33-(14-x)*1.5))<=1.1;
    if(inE(x,y,21,32,9.5,5.5)||inE(x,y,32,25,5.5,5)||tail)c=[72,72,84];      // body, head, tail
    if(y>=18&&y<=21&&(Math.abs(x-29)<=(y-17)*0.6||Math.abs(x-35)<=(y-17)*0.6))c=[72,72,84]; // ears
    if((x===30||x===34)&&y===24)c=[186,226,86];                               // eyes
    if(x===32&&y===27)c=[236,150,160];                                        // nose
    if(y>=36&&y<=37&&(x===15||x===16||x===26||x===27))c=[60,60,70];           // paws
    row.push(c);}px.push(row);}
  return px;
})();
const spt={p:8,sel:0};
const ptRGB=c=>'rgb('+c.join(',')+')';
function ptPatch(i){const n=PT_N/spt.p,r=Math.floor(i/n),c=i%n,px=[];for(let y=0;y<spt.p;y++)for(let x=0;x<spt.p;x++)px.push(PT_PIX[r*spt.p+y][c*spt.p+x]);return{r,c,px};}
function renderPatch(){
  const p=spt.p,n=PT_N/p,N=n*n,sel=ptPatch(spt.sel);let g='';
  PT_PIX.forEach((row,y)=>{let x=0;while(x<PT_N){let e=x;while(e+1<PT_N&&row[e+1].join()===row[x].join())e++;g+='<rect x="'+x+'" y="'+y+'" width="'+(e-x+1)+'" height="1" fill="'+ptRGB(row[x])+'" shape-rendering="crispEdges"/>';x=e+1;}});
  for(let k=1;k<n;k++)g+='<line class="ipg" x1="'+k*p+'" y1="0" x2="'+k*p+'" y2="48"/><line class="ipg" x1="0" y1="'+k*p+'" x2="48" y2="'+k*p+'"/>';
  g+='<rect class="ips o" x="'+sel.c*p+'" y="'+sel.r*p+'" width="'+p+'" height="'+p+'"/><rect class="ips" x="'+sel.c*p+'" y="'+sel.r*p+'" width="'+p+'" height="'+p+'"/>';
  const img=$('#pt-img');img.innerHTML=g;img.setAttribute('aria-label','A 48 by 48 pixel picture of a cat on a mat, cut into '+N+' patches. Selected: row '+(sel.r+1)+', column '+(sel.c+1)+'. Use the arrow keys to move.');
  $('#pt-zoom').setAttribute('viewBox','0 0 '+p+' '+p);
  $('#pt-zoom').innerHTML=sel.px.map((c,k)=>'<rect x="'+k%p+'" y="'+Math.floor(k/p)+'" width="1" height="1" fill="'+ptRGB(c)+'" shape-rendering="crispEdges"/>').join('');
  $('#pt-lab').textContent='Image token '+(spt.sel+1)+' of '+N+': row '+(sel.r+1)+', column '+(sel.c+1);
  const nums=[].concat(...sel.px.map(c=>c.map(v=>(v/255).toFixed(2))));
  $('#pt-nums').innerHTML=p+' × '+p+' pixels × 3 colours = '+nums.length.toLocaleString('en-US')+' numbers:<br>'+nums.slice(0,9).join(', ')+', …<br><small>red, green and blue for each pixel, row by row, scaled to 0–1. A learned matrix turns this row into one vector of width '+sym('d_model')+'.</small>';
  $('#pt-stats').innerHTML=[['Patch grid',n+' × '+n],['Image tokens',N],['Numbers per patch',(p*p*3).toLocaleString('en-US')],['Pairs attention compares',((N+PT_Q.length)**2).toLocaleString('en-US')]].map(r=>'<div><b>'+r[1]+'</b><span>'+r[0]+'</span></div>').join('');
  let s='';for(let i=0;i<N;i++){const px=ptPatch(i).px,avg=[0,1,2].map(k=>Math.round(px.reduce((a,c)=>a+c[k],0)/px.length));s+='<i data-i="'+i+'" class="'+(i===spt.sel?'on':'')+'" style="background:'+ptRGB(avg)+'"></i>';}
  $('#pt-seq').innerHTML=s+PT_Q.map(t=>'<span class="chip tok">'+esc(t.trim())+'<small>text</small></span>').join('');
  $$('#pt-size button').forEach(b=>{const on=+b.dataset.v===p;b.classList.toggle('on',on);b.setAttribute('aria-pressed',on);});
}
function initPatch(){
  $('#pt-size').innerHTML=[4,8,12,16,24].map(v=>'<button class="chip" data-v="'+v+'" aria-pressed="false">'+v+'</button>').join('');
  const at=(x,y)=>{const n=PT_N/spt.p;return Math.min(n-1,Math.floor(y/spt.p))*n+Math.min(n-1,Math.floor(x/spt.p));};
  spt.sel=at(32,24);
  $('#pt-size').addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;spt.p=+b.dataset.v;spt.sel=at(32,24);renderPatch();$('#pt-size button[data-v="'+spt.p+'"]').focus();});
  const img=$('#pt-img');
  img.addEventListener('click',e=>{const r=img.getBoundingClientRect();spt.sel=at((e.clientX-r.left)/r.width*PT_N,(e.clientY-r.top)/r.height*PT_N);renderPatch();});
  img.addEventListener('keydown',e=>{const n=PT_N/spt.p,d={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,-1],ArrowDown:[0,1]}[e.key];if(!d)return;e.preventDefault();
    const r=Math.max(0,Math.min(n-1,Math.floor(spt.sel/n)+d[1])),c=Math.max(0,Math.min(n-1,spt.sel%n+d[0]));spt.sel=r*n+c;renderPatch();});
  $('#pt-seq').addEventListener('click',e=>{const t=e.target.closest('i[data-i]');if(t){spt.sel=+t.dataset.i;renderPatch();}});
  reg(renderPatch);
}
/* ================= How diffusion models make images: noise to a spiral ================= */
/* Each example is a 2-D point. The denoiser is the exact best guess for the 300 training points (a weighted
   average of them), not a trained network; the widget says so. */
const DF_DATA=(()=>{const r=mulberry32(3),d=[];for(let i=0;i<300;i++){const t=1.5+Math.sqrt(r())*2.6*Math.PI,rad=t/(1.5+2.6*Math.PI)*1.3;d.push([rad*Math.cos(t)+0.025*randn(r),rad*Math.sin(t)+0.025*randn(r)]);}return d;})();
const DF_EPS=(()=>{const r=mulberry32(17);return DF_DATA.map(()=>[randn(r),randn(r)]);})();
const sdf={mode:'gen',S:50,k:50,pts:null,seed:1,timer:null,lev:0.5};
const dfAb=lev=>Math.max(1e-6,Math.cos(Math.PI/2*Math.min(lev,0.9999))**2);
function dfGuess(u,a){
  let m=-Infinity;const L=DF_DATA.map(d=>{const dx=u[0]-Math.sqrt(a)*d[0],dy=u[1]-Math.sqrt(a)*d[1],l=-(dx*dx+dy*dy)/(2*(1-a));if(l>m)m=l;return l;});
  let s=0,x=0,y=0;L.forEach((l,i)=>{const w=Math.exp(l-m);s+=w;x+=w*DF_DATA[i][0];y+=w*DF_DATA[i][1];});return[x/s,y/s];
}
function dfReset(){const r=mulberry32(sdf.seed);sdf.rng=r;sdf.pts=DF_DATA.map(()=>[randn(r),randn(r)]);sdf.k=sdf.S;}
/* one step from level k/S to (k-1)/S: guess the clean point, move part of the way towards it, add a little fresh noise */
function dfStep(){
  if(sdf.k<=0)return;
  const a=dfAb(sdf.k/sdf.S),a2=sdf.k>1?dfAb((sdf.k-1)/sdf.S):1,sig=Math.sqrt((1-a2)/(1-a)*Math.max(0,1-a/a2)),c=Math.sqrt(Math.max(0,1-a2-sig*sig));
  sdf.pts=sdf.pts.map(u=>{const g=dfGuess(u,a);return[0,1].map(j=>Math.sqrt(a2)*g[j]+c*(u[j]-Math.sqrt(a)*g[j])/Math.sqrt(1-a)+sig*randn(sdf.rng));});
  sdf.k--;
}
const dfNear=p=>{let b=Infinity;DF_DATA.forEach(d=>{const v=Math.hypot(p[0]-d[0],p[1]-d[1]);if(v<b)b=v;});return b;};
function renderDiff(){
  const X=v=>(120+v*50).toFixed(1),Y=v=>(120-v*50).toFixed(1),cl=v=>Math.max(-2.3,Math.min(2.3,v));
  const gen=sdf.mode==='gen',lev=gen?sdf.k/sdf.S:sdf.lev,a=dfAb(lev);
  const pts=gen?sdf.pts:DF_DATA.map((d,i)=>[0,1].map(j=>Math.sqrt(a)*d[j]+Math.sqrt(1-a)*DF_EPS[i][j]));
  let g=DF_DATA.map(d=>'<circle class="dfd" cx="'+X(d[0])+'" cy="'+Y(d[1])+'" r="1.3"/>').join('');
  g+=pts.map(p=>'<circle class="dfp" cx="'+X(cl(p[0]))+'" cy="'+Y(cl(p[1]))+'" r="1.9"/>').join('');
  const h=pts[0];g+='<circle class="dfr" cx="'+X(cl(h[0]))+'" cy="'+Y(cl(h[1]))+'" r="5"/>';
  if(lev>0.001){const q=dfGuess(h,a);g+='<line class="dfl" x1="'+X(cl(h[0]))+'" y1="'+Y(cl(h[1]))+'" x2="'+X(q[0])+'" y2="'+Y(q[1])+'"/><path class="dfx" d="M'+(+X(q[0])-3.5)+' '+(+Y(q[1])-3.5)+'l7 7m0 -7l-7 7"/>';}
  $('#df-svg').innerHTML=g;
  const on=pts.filter(p=>dfNear(p)<0.03).length,pc=v=>Math.round(100*v)+'%';
  /* covered: training points with a sample near them, so that samples piled on one spot do not count as success */
  const cov=()=>DF_DATA.filter(d=>pts.some(p=>Math.hypot(p[0]-d[0],p[1]-d[1])<0.05)).length/DF_DATA.length;
  $('#df-stats').innerHTML=(gen?[['Step',(sdf.S-sdf.k)+' / '+sdf.S],['Noise level',fmt(lev,2)],['Weight on the original',fmt(Math.sqrt(a),2)],['Samples on the spiral',pc(on/pts.length)],['Spiral covered by samples',pc(cov())]]
    :[['Noise level',fmt(lev,2)],['Weight on the original',fmt(Math.sqrt(a),2)],['Points still on the spiral',pc(on/pts.length)]]).map(r=>'<div><b>'+r[1]+'</b><span>'+r[0]+'</span></div>').join('');
  $('#df-play').textContent=sdf.timer?'❚❚ Pause':sdf.k<=0?'↺ Generate again':'▶ Generate';
  $('#df-steps-c').hidden=$('#df-btns').hidden=!gen;$('#df-lev-c').hidden=gen;
  [['df-mode',sdf.mode],['df-steps',String(sdf.S)]].forEach(([id,v])=>$$('#'+id+' button').forEach(b=>{const o=b.dataset.v===v;b.classList.toggle('on',o);b.setAttribute('aria-pressed',o);}));
}
function initDiff(){
  dfReset();
  const stop=()=>{if(sdf.timer){clearInterval(sdf.timer);sdf.timer=null;}};
  const chips=(id,opts,set)=>{const el=$('#'+id);el.innerHTML=opts.map(o=>'<button class="chip" data-v="'+o[0]+'" aria-pressed="false">'+o[1]+'</button>').join('');
    el.addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;stop();set(b.dataset.v);renderDiff();$('#'+id+' button[data-v="'+b.dataset.v+'"]').focus();});};
  chips('df-mode',[['gen','Remove noise (generate)'],['fwd','Add noise (training)']],v=>{sdf.mode=v;});
  chips('df-steps',[['1','1'],['3','3'],['10','10'],['50','50']],v=>{sdf.S=+v;dfReset();});
  slider('df-lev',v=>v.toFixed(2),v=>{sdf.lev=v;renderDiff();});
  $('#df-play').addEventListener('click',()=>{
    if(sdf.timer){stop();renderDiff();return;}
    if(sdf.k<=0)dfReset();
    sdf.timer=setInterval(()=>{dfStep();if(sdf.k<=0)stop();renderDiff();},Math.max(60,1500/sdf.S));renderDiff();});
  $('#df-step').addEventListener('click',()=>{stop();if(sdf.k<=0)dfReset();else dfStep();renderDiff();});
  $('#df-new').addEventListener('click',()=>{stop();sdf.seed++;dfReset();renderDiff();});
  reg(renderDiff);
}

/* ================= Mixture of experts: route tokens to experts ================= */
/* Token vectors: a base vector per kind of token plus a fixed per-token offset, so the same token always gets the
   same vector. The router is a random matrix (not trained) plus one bias per expert. */
const MOE_E=8,MOE_D=6,MOE_LINES=[
  ['the','cat','sat','on','the','mat','.','the','dog','slept','on','the','sofa','.'],
  ['def','add','(','a',',','b',')',':','return','a','+','b'],
  ['in','2017',',','8','heads','and','6','layers','were','used','.']];
const MOE_KIND={the:0,on:0,and:0,in:0,were:0,'.':1,',':1,'(':1,')':1,':':1,'+':1,'2017':2,'8':2,'6':2,def:3,return:3,a:3,b:3,add:3};
const MOE=(()=>{
  const r=mulberry32(41),vec=()=>Array.from({length:MOE_D},()=>randn(r));
  const base=[vec(),vec(),vec(),vec(),vec()],W=Array.from({length:MOE_D},()=>Array.from({length:MOE_E},()=>randn(r)*0.8)),cache={};
  const emb=t=>{if(cache[t])return cache[t];let h=7;for(const ch of t)h=(h*31+ch.charCodeAt(0))>>>0;const rr=mulberry32(h),k=MOE_KIND[t]!=null?MOE_KIND[t]:4;
    return cache[t]=base[k].map(v=>v+0.6*randn(rr));};
  const toks=[].concat(...MOE_LINES.map((l,li)=>l.map(t=>({t,li}))));
  toks.forEach(o=>{const x=emb(o.t);o.score=Array.from({length:MOE_E},(_,e)=>x.reduce((a,v,d)=>a+v*W[d][e],0));});
  return{toks};
})();
const smoe={k:2,router:'init',sel:3,bias:null};
function moeBias(kind){
  const b=Array(MOE_E).fill(0);
  if(kind==='fav')b[4]=3;
  if(kind==='bal')for(let it=0;it<200;it++){const load=moeLoad(b,smoe.k),avg=load.reduce((a,v)=>a+v,0)/MOE_E;load.forEach((l,e)=>{b[e]-=0.1*Math.pow(0.98,it)*Math.sign(l-avg);});}
  return b;
}
function moeRoute(o,b,k){
  const s=o.score.map((v,e)=>v+b[e]),ord=s.map((v,e)=>e).sort((x,y)=>s[y]-s[x]),top=ord.slice(0,k),m=Math.max(...top.map(e=>s[e])),ex=top.map(e=>Math.exp(s[e]-m)),z=ex.reduce((a,v)=>a+v,0);
  return{s,top,gate:top.map((e,i)=>ex[i]/z)};
}
function moeLoad(b,k){const L=Array(MOE_E).fill(0);MOE.toks.forEach(o=>moeRoute(o,b,k).top.forEach(e=>L[e]++));return L;}
const moeCol=e=>viridis(0.22+0.78*e/(MOE_E-1));
function renderMoe(){
  const b=moeBias(smoe.router),k=smoe.k,routes=MOE.toks.map(o=>moeRoute(o,b,k));
  let h='',li=-1;
  MOE.toks.forEach((o,i)=>{if(o.li!==li){if(li>=0)h+='</div>';li=o.li;h+='<div class="moel">';}
    const R=routes[i];h+='<button type="button" class="chip tok moet'+(i===smoe.sel?' on':'')+'" data-i="'+i+'" aria-pressed="'+(i===smoe.sel)+'">'+esc(o.t)+'<small>'+R.top.map((e,j)=>'<i style="background:'+moeCol(e)+';opacity:'+(0.35+0.65*R.gate[j]).toFixed(2)+'" title="expert '+(e+1)+'"></i>').join('')+'</small></button>';});
  $('#moe-toks').innerHTML=h+'</div>';
  const o=MOE.toks[smoe.sel],R=routes[smoe.sel],m=Math.max(...R.s),pall=R.s.map(v=>Math.exp(v-m)),z=pall.reduce((a,v)=>a+v,0);
  $('#moe-lab').textContent='Router scores for “'+o.t+'”: '+(k===1?'the top expert is chosen, with a gate of 1':'the top '+k+' experts are chosen, and their gates are a softmax over the chosen scores only');
  $('#moe-bars').innerHTML=bars(R.s.map((v,e)=>{const j=R.top.indexOf(e);return{label:'<span class="moesw" style="background:'+moeCol(e)+'"></span>expert '+(e+1),p:pall[e]/z,cls:j>=0?'hit':'',right:'score '+fmt(v,2)+(j>=0?' · gate '+fmt(R.gate[j],2):''),tip:j>=0?'Chosen: this expert’s output is multiplied by '+fmt(R.gate[j],3):'Not chosen: this expert does no work for this token'};}),1);
  const load=moeLoad(b,k),avg=load.reduce((a,v)=>a+v,0)/MOE_E;
  $('#moe-load').innerHTML=bars(load.map((l,e)=>({label:'<span class="moesw" style="background:'+moeCol(e)+'"></span>expert '+(e+1),p:l,right:l+' tokens',tip:'Expert '+(e+1)+' processes '+l+' of the '+MOE.toks.length+' tokens'})),Math.max(...load));
  const shared=1.63,per=5.63,act=shared+k*per;
  $('#moe-stats').innerHTML=[['Experts used per token',k+' of '+MOE_E],['Busiest expert’s load',fmt(Math.max(...load)/avg,1)+'× the average'],['Experts with no tokens',load.filter(l=>!l).length],['Active parameters, Mixtral-shaped',fmt(act,1)+'B of 46.7B']].map(r=>'<div><b>'+r[1]+'</b><span>'+r[0]+'</span></div>').join('');
  [['moe-k',String(k)],['moe-r',smoe.router]].forEach(([id,v])=>$$('#'+id+' button').forEach(bt=>{const on=bt.dataset.v===v;bt.classList.toggle('on',on);bt.setAttribute('aria-pressed',on);}));
}
function initMoe(){
  const chips=(id,opts,set)=>{const el=$('#'+id);el.innerHTML=opts.map(o=>'<button class="chip" data-v="'+o[0]+'" aria-pressed="false">'+o[1]+'</button>').join('');
    el.addEventListener('click',e=>{const bt=e.target.closest('button');if(!bt)return;set(bt.dataset.v);renderMoe();$('#'+id+' button[data-v="'+bt.dataset.v+'"]').focus();});};
  chips('moe-k',[['1','1'],['2','2'],['4','4']],v=>{smoe.k=+v;});
  chips('moe-r',[['init','As initialized'],['fav','With a favourite expert'],['bal','Balanced']],v=>{smoe.router=v;});
  $('#moe-toks').addEventListener('click',e=>{const bt=e.target.closest('.moet');if(bt){smoe.sel=+bt.dataset.i;renderMoe();$('#moe-toks .moet[data-i="'+smoe.sel+'"]').focus();}});
  reg(renderMoe);
}

/* ================= Retrieval: keyword, vector and hybrid search over a small handbook ================= */
const RG_DOCS=['Reset your password from the account settings page.',
  'Forgotten password? Use the self-service reset link on the sign-in page.',
  'Expense claims must be filed within 30 days of purchase, with receipts.',
  'Annual leave requests go to your line manager for approval.',
  'Update the VPN client before connecting from home.',
  'Error E-1042 on the VPN client means your certificate has expired: ask the service desk for a new one.',
  'Laptops are replaced every four years; report damage to the service desk.',
  'Flights and hotels over 500 need approval from finance before booking.'];
const RG_DIMS=['accounts','money','leave','network','devices','time','approval','problems'];
/* a hand-made embedding, not a trained one: each known word adds weight to one or more of eight meanings */
const RG_EMB={password:[[0,1]],reset:[[0,.7],[7,.3]],account:[[0,1]],'sign-in':[[0,.9]],login:[[0,.9]],log:[[0,.6]],
  forgot:[[0,.5],[7,.5]],forgotten:[[0,.5],[7,.5]],locked:[[0,.6],[7,.5]],'self-service':[[0,.3]],link:[[0,.2]],settings:[[0,.5],[4,.3]],
  expense:[[1,1]],expenses:[[1,1]],claims:[[1,.8]],claim:[[1,.8]],money:[[1,1]],spent:[[1,.8]],spend:[[1,.8]],purchase:[[1,.8]],bought:[[1,.8]],
  receipts:[[1,.7]],refund:[[1,.8]],cost:[[1,.8]],paid:[[1,.8]],'500':[[1,.4]],flights:[[1,.5],[6,.3]],hotels:[[1,.5],[6,.3]],booking:[[1,.4]],
  travel:[[1,.5]],finance:[[1,.8],[6,.3]],leave:[[2,1]],holiday:[[2,1]],holidays:[[2,1]],vacation:[[2,1]],annual:[[2,.6],[5,.3]],off:[[2,.5]],
  manager:[[6,.9]],approval:[[6,1]],approve:[[6,1]],requests:[[6,.4]],request:[[6,.4]],
  vpn:[[3,1]],network:[[3,1]],wifi:[[3,1]],internet:[[3,1]],connect:[[3,.8]],connecting:[[3,.8]],home:[[3,.4]],remote:[[3,.6]],client:[[3,.4],[4,.3]],
  certificate:[[3,.5],[7,.3],[0,.3]],update:[[4,.5]],laptop:[[4,1]],laptops:[[4,1]],computer:[[4,1]],screen:[[4,.8]],replaced:[[4,.5]],replace:[[4,.5]],
  new:[[4,.3]],damage:[[4,.5],[7,.6]],broken:[[4,.4],[7,.8]],desk:[[4,.3],[7,.2]],
  days:[[5,.8]],within:[[5,.8]],long:[[5,.6]],deadline:[[5,1]],years:[[5,.6]],when:[[5,.4]],late:[[5,.6],[7,.3]],
  error:[[7,1]],problem:[[7,.9]],fails:[[7,.8]],failed:[[7,.8]],wrong:[[7,.7]],expired:[[7,.5],[5,.4]]};
const RG_STOP=new Set('a an the i my me to do does how what of on in for from your you is are be it and or with have has can must go get before after over back this that these at by'.split(' '));
const RG_QS=['how do I reset my password','I forgot my login','how long do I have to claim back money I spent','E-1042 on my laptop'];
const RG_TOP=5;   /* each search returns its best 5 */
const rgTok=t=>(t.toLowerCase().match(/[a-z0-9]+(?:-[a-z0-9]+)*/g)||[]).filter(w=>!RG_STOP.has(w));
const RG_T=RG_DOCS.map(rgTok),RG_AVG=RG_T.reduce((a,d)=>a+d.length,0)/RG_T.length;
function rgEmbed(toks){const v=new Array(RG_DIMS.length).fill(0);toks.forEach(w=>(RG_EMB[w]||[]).forEach(([d,x])=>{v[d]+=x;}));
  const n=Math.hypot(...v);return n?v.map(x=>x/n):null;}
const RG_V=RG_T.map(rgEmbed);
/* BM25 with the usual settings k1 = 1.2 and b = 0.75 */
function rgBm25(q){const terms=[...new Set(q)];return RG_T.map(d=>terms.reduce((s,w)=>{const f=d.filter(x=>x===w).length;if(!f)return s;
  const n=RG_T.filter(x=>x.includes(w)).length,idf=Math.log(1+(RG_T.length-n+0.5)/(n+0.5));
  return s+idf*f*2.2/(f+1.2*(0.25+0.75*d.length/RG_AVG));},0));}
/* ranks 1, 2, … for the best RG_TOP scores that pass keep(); 0 for the rest */
function rgRank(sc,keep){const r=new Array(sc.length).fill(0);
  sc.map((s,i)=>[s,i]).filter(p=>keep(p[0])).sort((a,b)=>b[0]-a[0]||a[1]-b[1]).slice(0,RG_TOP).forEach((p,k)=>{r[p[1]]=k+1;});return r;}
function rgSearch(text){const q=rgTok(text),qv=rgEmbed(q),kw=rgBm25(q),cos=RG_V.map(v=>qv&&v?v.reduce((a,x,i)=>a+x*qv[i],0):0);
  const rk=rgRank(kw,s=>s>0),rv=qv?rgRank(cos,()=>true):new Array(RG_DOCS.length).fill(0);
  const fused=RG_DOCS.map((_,i)=>(rk[i]?1/(60+rk[i]):0)+(rv[i]?1/(60+rv[i]):0));
  return{q,qv,kw,cos,rk,rv,fused};}
const srg={text:RG_QS[1],k:2};
function renderRag(){
  const R=rgSearch(srg.text),order=RG_DOCS.map((_,i)=>i).filter(i=>R.fused[i]>0).sort((a,b)=>R.fused[b]-R.fused[a]||a-b),top=order.slice(0,srg.k);
  $('#rg-toks').innerHTML='Words searched for: '+(R.q.length?R.q.map(w=>'<span class="chip tok" style="cursor:default">'+esc(w)+(RG_EMB[w]?'':'<small>not in the embedding</small>')+'</span>').join(' '):'none (all were common words)');
  $('#rg-vec').innerHTML=R.qv?bars(RG_DIMS.map((d,i)=>({label:d,p:R.qv[i],right:fmt(R.qv[i],2)})),1):'<p class="cap">The embedding knows none of these words, so vector search has nothing to go on.</p>';
  const cell=(s,r,n)=>r?fmt(s,n)+' <small>#'+r+'</small>':'–';
  $('#rg-tab').innerHTML='<thead><tr><th>Chunk</th><th>Keyword (BM25)</th><th>Vector (cosine)</th><th>Fused (RRF)</th></tr></thead><tbody>'+
    order.map(i=>{const c=top.includes(i)?' cur':'',td=(h,k)=>'<td class="'+(k||'')+c+'">'+h+'</td>';
      return '<tr>'+td(esc(RG_DOCS[i]),'rgc')+td(cell(R.kw[i],R.rk[i],2))+td(cell(R.cos[i],R.rv[i],3))+td(fmt(R.fused[i],4))+'</tr>';}).join('')+'</tbody>';
  $('#rg-prompt').innerHTML=top.length?'Answer using only the sources below, and cite them by number. If they do not contain the answer, say so.<br><br>'+
    top.map((i,k)=>'['+(k+1)+'] '+esc(RG_DOCS[i])).join('<br>')+'<br><br>Question: '+esc(srg.text):'Nothing was found, so the prompt would have no sources. A good system says so instead of guessing.';
  $$('#rg-q button').forEach(b=>{const o=RG_QS[+b.dataset.v]===srg.text;b.classList.toggle('on',o);b.setAttribute('aria-pressed',o);});
  $$('#rg-k button').forEach(b=>{const o=+b.dataset.v===srg.k;b.classList.toggle('on',o);b.setAttribute('aria-pressed',o);});
}
function initRag(){
  $('#rg-q').innerHTML=RG_QS.map((q,i)=>'<button class="chip" data-v="'+i+'" aria-pressed="false">'+esc(q)+'</button>').join('');
  $('#rg-q').addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;srg.text=RG_QS[+b.dataset.v];$('#rg-in').value=srg.text;renderRag();b.focus();});
  $('#rg-k').innerHTML=[2,3].map(k=>'<button class="chip" data-v="'+k+'" aria-pressed="false">'+k+'</button>').join('');
  $('#rg-k').addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;srg.k=+b.dataset.v;renderRag();b.focus();});
  $('#rg-in').value=srg.text;
  $('#rg-in').addEventListener('input',e=>{srg.text=e.target.value;renderRag();});
  reg(renderRag);
}

PAGE_INITS.push(initCross,initCrossKeys,initLR,initRope,initGqa,initMoe,initPatch,initDiff,initRag);
