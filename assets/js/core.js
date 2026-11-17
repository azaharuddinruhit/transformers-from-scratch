'use strict';
/* Shared by every page: DOM helpers, the render registry, number formatting, maths rendering, colour,
   shared chart components, tooltips, the Python highlighter and the themed dropdowns. */
const $=(s,r)=>(r||document).querySelector(s),$$=(s,r)=>Array.from((r||document).querySelectorAll(s));
const widgets=[];
function reg(fn){widgets.push(fn);try{fn();}catch(e){console.error(e);}}
/* widget init functions this page runs, in order (each part's file adds its own) */
const PAGE_INITS=[];
function rerender(){widgets.forEach(f=>{try{f();}catch(e){console.error(e);}});}
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function fmt(x,n){if(x===-Infinity)return '−∞';if(!isFinite(x))return '∞';if(n==null)n=4;let s=x.toFixed(n);if(parseFloat(s)===0)s=(0).toFixed(n);return s.replace('-','−');}
function big(n){const a=Math.abs(n);if(a>=1e12)return (n/1e12).toFixed(2)+' T';if(a>=1e9)return (n/1e9).toFixed(2)+' B';if(a>=1e6)return (n/1e6).toFixed(2)+' M';if(a>=1e3)return (n/1e3).toFixed(1)+' K';return String(Math.round(n));}
function bytes(b){if(b>=1e12)return (b/1e12).toFixed(2)+' TB';if(b>=1e9)return (b/1e9).toFixed(2)+' GB';if(b>=1e6)return (b/1e6).toFixed(1)+' MB';if(b>=1e3)return (b/1e3).toFixed(1)+' KB';return Math.round(b)+' B';}
const SUP={'-':'⁻','0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹'};
const pow10=n=>'10'+String(n).split('').map(c=>SUP[c]||c).join('');
function sci(x){if(!isFinite(x))return '∞';if(x===0)return '0';const e=Math.floor(Math.log10(Math.abs(x)));if(e>-3&&e<4)return x.toPrecision(3);return (x/Math.pow(10,e)).toFixed(1)+'×'+pow10(e);}
const showSp=c=>c===' '?'␣':c;
function slider(id,f,cb){const e=$('#'+id),o=$('#'+id+'-o'),u=()=>{const t=String(f(+e.value));if(o)o.textContent=t;e.setAttribute('aria-valuetext',t);};e.addEventListener('input',()=>{u();cb(+e.value);});u();return e;}

/* ---------- maths rendering (KaTeX with plain-text fallback) ---------- */
/* d_model -> d<sub>model</sub> for text that widgets rewrite (KaTeX is used for static text) */
const sym=t=>t.replace(/\b([dnTXEW])_(model|ff|vocab|src|tgt|enc|dec|pos|[kQKVOU])\b/g,(m,a,b)=>'<span class="msym"><i>'+a+'</i><sub>'+(b.length===1?'<i>'+b+'</i>':b)+'</sub></span>');
function renderTex(root){if(!window.katex)return;$$('.tex',root).forEach(e=>{if(e.dataset.done)return;const src=e.dataset.src||e.textContent;e.dataset.src=src;try{window.katex.render(src,e,{displayMode:e.classList.contains('d'),throwOnError:false});e.dataset.done='1';}catch(err){e.textContent=src;}});}
window.renderAllTex=function(){renderTex(document);};
window.texFailed=function(){$('#texwarn').hidden=false;};

/* ---------- colour ---------- */
const VIR=[[68,1,84],[72,40,120],[62,74,137],[49,104,142],[38,130,142],[31,158,137],[53,183,121],[109,205,89],[180,222,44],[253,231,37]];
function viridis(t){t=Math.max(0,Math.min(1,t))*(VIR.length-1);const i=Math.min(VIR.length-2,Math.floor(t)),f=t-i;return 'rgb('+VIR[i].map((c,k)=>Math.round(c+(VIR[i+1][k]-c)*f)).join(',')+')';}
const vtext=t=>t>0.62?'#111':'#fff';
function divColor(v){const t=Math.max(-1,Math.min(1,v)),a=t<0?[33,102,172]:[230,97,1],n=[244,241,234],k=Math.abs(t);return 'rgb('+a.map((c,i)=>Math.round(n[i]+(c-n[i])*k)).join(',')+')';}

/* ---------- shared components ---------- */
function vec(vals,kind,n,mx){mx=mx||Math.max(1e-9,Math.max.apply(null,vals.map(Math.abs)));return '<span class="vec '+(kind||'')+'">'+vals.map(v=>'<span class="c" style="--a:'+(0.06+0.5*Math.min(1,Math.abs(v)/mx)).toFixed(2)+'">'+fmt(v,n)+'</span>').join('')+'</span>';}
/* bar fills take their colour from their value (viridis); a cut bar keeps its hatching */
function bars(rows,max){max=max||Math.max(1e-9,Math.max.apply(null,rows.map(r=>r.p)));return '<div class="bars">'+rows.map(r=>'<div class="bar-row '+(r.cls||'')+'"'+(r.tip?' tabindex="0" data-tip="'+esc(r.tip)+'"':'')+'><span class="bl">'+r.label+'</span><span class="bt"><span class="bf" style="width:'+(100*Math.max(0,r.p)/max).toFixed(1)+'%'+(/cut/.test(r.cls||'')?'':';background:'+viridis(.25+.7*Math.max(0,r.p)/max))+'"></span></span><span class="bv">'+(r.right!=null?r.right:fmt(r.p))+'</span></div>').join('')+'</div>';}
function heat(M,rl,cl,o){o=o||{};let h='<div class="hm'+(o.compact?' compact':'')+'" style="grid-template-columns:auto repeat('+cl.length+',minmax(0,1fr))"><span></span>'+cl.map(c=>'<span class="hh">'+c+'</span>').join('');
  M.forEach((row,i)=>{h+='<span class="hr'+(o.sel===i?' on':'')+'">'+rl[i]+'</span>';row.forEach((v,j)=>{const mk=o.mask&&o.mask(i,j);const tp=o.tip?o.tip(i,j,v,mk):fmt(v,4);h+='<span class="hc'+(mk?' masked':'')+(o.sel===i?' selrow':'')+'"'+(o.compact?'':' tabindex="0" role="'+(o.click?'button':'img')+'" aria-label="'+esc(tp.replace(/<br>/g,', ').replace(/<[^>]+>/g,'').replace(/&gt;/g,'greater than').replace(/&lt;/g,'less than'))+'"')+' data-i="'+i+'" data-j="'+j+'" style="background-color:'+viridis(v)+';color:'+vtext(v)+'" data-tip="'+esc(tp)+'">'+(o.compact?Math.round(v*100):fmt(v,2))+'</span>';});});
  return h+'</div>';}
function lineChart(svg,series,o){
  const W=Math.max(240,Math.round(svg.getBoundingClientRect().width)||560),H=o.h||250,m={l:o.ml||46,r:14,t:12,b:38};
  svg.setAttribute('viewBox','0 0 '+W+' '+H);svg.style.height=H+'px';
  const cl=(v,a,b)=>Math.max(a,Math.min(b,isNaN(v)?b:v));
  const X=x=>m.l+(x-o.x0)/(o.x1-o.x0)*(W-m.l-m.r),Y=y=>H-m.b-(cl(y,o.y0,o.y1)-o.y0)/(o.y1-o.y0)*(H-m.t-m.b);
  let s='';
  (o.yt||[]).forEach(v=>{s+='<line class="grid" x1="'+m.l+'" x2="'+(W-m.r)+'" y1="'+Y(v)+'" y2="'+Y(v)+'"/><text class="ax" x="'+(m.l-6)+'" y="'+(Y(v)+4)+'" text-anchor="end">'+(o.yf?o.yf(v):v)+'</text>';});
  (o.xt||[]).forEach(v=>{s+='<text class="ax" x="'+X(v)+'" y="'+(H-m.b+15)+'" text-anchor="middle">'+(o.xf?o.xf(v):v)+'</text>';});
  if(o.xl)s+='<text class="ax" x="'+(m.l+W-m.r)/2+'" y="'+(H-5)+'" text-anchor="middle">'+o.xl+'</text>';
  if(o.yl)s+='<text class="ax" transform="rotate(-90)" x="'+(-(m.t+H-m.b)/2)+'" y="11" text-anchor="middle">'+o.yl+'</text>';
  series.forEach(se=>{if(se.pts.length>1)s+='<polyline class="ln '+se.cls+'" points="'+se.pts.map(p=>X(p[0]).toFixed(1)+','+Y(p[1]).toFixed(1)).join(' ')+'"/>';
    /* an optional label at the line's end, at height ly when given (to keep labels apart) */
    if(se.label&&se.pts.length){const p=se.pts[se.pts.length-1];s+='<text class="ax" x="'+(X(p[0])-4)+'" y="'+(Y(se.ly!=null?se.ly:p[1])-8)+'" text-anchor="end" paint-order="stroke" stroke="var(--surface)" stroke-width="3">'+se.label+'</text>';}});
  (o.dots||[]).forEach(d=>{const x=X(d[0]),y=Y(d[1]),end=x>W*0.65;s+='<circle class="dotm" cx="'+x+'" cy="'+y+'" r="6"/>'+(d[2]?'<text class="ax" style="font-weight:700" x="'+(end?x-10:x+10)+'" y="'+(y-9)+'" text-anchor="'+(end?'end':'start')+'">'+d[2]+'</text>':'');});
  svg.innerHTML=s;
}
function signedBars(svg,vals,ymax){
  const W=Math.max(200,Math.round(svg.getBoundingClientRect().width)||300),H=170,mid=H/2,bw=(W-10)/vals.length;
  svg.setAttribute('viewBox','0 0 '+W+' '+H);svg.style.height=H+'px';
  let s='<line x1="0" x2="'+W+'" y1="'+mid+'" y2="'+mid+'" stroke="var(--muted)"/>';
  vals.forEach((v,i)=>{const h=Math.min(1,Math.abs(v)/ymax)*(mid-20),x=5+i*bw+3,a=Math.abs(v);s+='<rect x="'+x+'" y="'+(v>=0?mid-h:mid)+'" width="'+(bw-6)+'" height="'+Math.max(1,h)+'" rx="3" fill="var(--accent)"/><text class="ax" x="'+(x+(bw-6)/2)+'" y="'+(v>=0?mid-h-4:mid+h+12)+'" text-anchor="middle">'+fmt(v,a>=100?0:a>=10?1:2)+'</text>';});
  svg.innerHTML=s;
}

/* ---------- tooltips ---------- */
const tipEl=$('#tip');
function hideTip(){tipEl.style.display='none';}
function tipFor(t){tipEl.innerHTML=t.getAttribute('data-tip');tipEl.style.display='block';const r=t.getBoundingClientRect(),b=tipEl.getBoundingClientRect();let top=r.top-b.height-8;if(top<60)top=r.bottom+8;tipEl.style.left=Math.min(window.innerWidth-b.width-8,Math.max(8,r.left+r.width/2-b.width/2))+'px';tipEl.style.top=top+'px';}
document.addEventListener('mouseover',e=>{const t=e.target.closest&&e.target.closest('[data-tip]');if(t)tipFor(t);else hideTip();});
document.addEventListener('focusin',e=>{const t=e.target.closest&&e.target.closest('[data-tip]');if(t)tipFor(t);else hideTip();});
document.addEventListener('focusout',hideTip);
window.addEventListener('scroll',hideTip,{passive:true});

/* Enter or Space on a focused heatmap cell selects it, and focus stays on the redrawn cell */
function gridKeys(sel){$(sel).addEventListener('keydown',e=>{if(e.key!=='Enter'&&e.key!==' ')return;const c=e.target.closest('.hc');if(!c)return;e.preventDefault();const i=c.dataset.i,j=c.dataset.j;c.click();const n=$(sel+' .hc[data-i="'+i+'"][data-j="'+j+'"]');if(n)n.focus();});}

function initCode(){
  const KW=new Set('False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield'.split(' '));
  const BI=new Set('print range len sum min max int float str list dict set tuple enumerate zip sorted round abs isinstance super next iter any all map repr type Exception ValueError RuntimeError'.split(' '));
  const re=/(#[^\n]*)|((?:[rRbBfFuU]{1,2})?(?:"""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'))|(@[A-Za-z_][\w.]*)|(\b\d[\d_]*(?:\.\d*)?(?:[eE][+-]?\d+)?\b)|([A-Za-z_]\w*)|([-+*/%=<>!&|^~@]+)/g;
  const esc=t=>t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  /* classes: c comment, s string, d decorator, n number, k keyword, f function (defined or called), b builtin, sf self,
     t class name (capital then a lower-case letter, so T and F stay plain), p keyword argument (name= inside brackets), o operator */
  $$('pre.py code').forEach(code=>{
    const src=code.textContent;let out='',last=0,prev='',depth=0,m;re.lastIndex=0;
    while((m=re.exec(src))){
      const t=m[0],gap=src.slice(last,m.index),after=src.slice(re.lastIndex,re.lastIndex+40);let c='';
      for(const ch of gap){if('([{'.includes(ch))depth++;else if(')]}'.includes(ch))depth=Math.max(0,depth-1);}
      if(m[1])c='c';else if(m[2])c='s';else if(m[3])c='d';else if(m[4])c='n';else if(m[6])c='o';
      else if(KW.has(t))c='k';else if(t==='self')c='sf';else if(prev==='class')c='t';else if(prev==='def')c='f';
      else if(depth&&/^\s*=(?!=)/.test(after))c='p';
      else if(BI.has(t)&&src[m.index-1]!=='.')c='b';
      else if(/^[A-Z]\w*[a-z]/.test(t))c='t';
      else if(/^\s*\(/.test(after))c='f';
      out+=esc(gap)+(c?'<span class="'+c+'">'+esc(t)+'</span>':esc(t));
      last=re.lastIndex;prev=m[5]?t:'';
    }
    code.innerHTML=out+esc(src.slice(last));
  });
  $$('.cpy').forEach(b=>b.addEventListener('click',()=>{
    const text=b.closest('details').querySelector('pre code').textContent;
    const done=()=>{b.classList.add('ok');b.setAttribute('aria-label','Copied');setTimeout(()=>{b.classList.remove('ok');b.setAttribute('aria-label','Copy code');},1500);};
    const fallback=()=>{const ta=document.createElement('textarea');ta.value=text;ta.style.cssText='position:fixed;opacity:0';document.body.appendChild(ta);ta.select();try{document.execCommand('copy');done();}catch(e){}ta.remove();};
    if(navigator.clipboard&&window.isSecureContext)navigator.clipboard.writeText(text).then(done,fallback);else fallback();
  }));
}

/* ================= themed dropdowns ================= */
function initSelects(){
  let uid=0,openDD=null;
  const closeAll=()=>{if(openDD)openDD.close();};
  document.addEventListener('click',e=>{if(openDD&&!openDD.wrap.contains(e.target))closeAll();});
  document.addEventListener('focusin',e=>{if(openDD&&!openDD.wrap.contains(e.target))closeAll();});
  $$('select').forEach(sel=>{
    const id='dd'+(++uid),wrap=document.createElement('div');wrap.className='dd';
    const lab=sel.closest('label'),labText=sel.getAttribute('aria-label')||(lab?Array.from(lab.childNodes).filter(n=>n.nodeType===3).map(n=>n.textContent).join(' ').trim():'');
    const btn=document.createElement('button');btn.type='button';btn.className='dd-btn';btn.setAttribute('aria-haspopup','listbox');btn.setAttribute('aria-expanded','false');btn.setAttribute('aria-controls',id+'-l');
    const txt=document.createElement('span');btn.appendChild(txt);
    const list=document.createElement('ul');list.className='dd-list';list.id=id+'-l';list.setAttribute('role','listbox');list.tabIndex=-1;list.hidden=true;if(labText)list.setAttribute('aria-label',labText);
    const opts=Array.from(sel.options);
    list.innerHTML=opts.map((o,i)=>'<li class="dd-opt" role="option" id="'+id+'-o'+i+'" data-i="'+i+'">'+esc(o.textContent)+'</li>').join('');
    wrap.appendChild(btn);wrap.appendChild(list);
    sel.classList.add('dd-native');sel.tabIndex=-1;sel.setAttribute('aria-hidden','true');sel.insertAdjacentElement('afterend',wrap);
    let act=sel.selectedIndex;
    const items=()=>Array.from(list.children);
    const paint=()=>{const cur=sel.selectedIndex;txt.textContent=cur>=0?sel.options[cur].textContent:'';btn.setAttribute('aria-label',(labText?labText+': ':'')+txt.textContent);items().forEach((li,i)=>li.setAttribute('aria-selected',i===cur));};
    const setAct=i=>{act=Math.max(0,Math.min(opts.length-1,i));items().forEach((li,j)=>li.classList.toggle('act',j===act));list.setAttribute('aria-activedescendant',id+'-o'+act);const li=items()[act];if(li&&li.scrollIntoView)li.scrollIntoView({block:'nearest'});};
    const choose=i=>{if(i!==sel.selectedIndex){sel.selectedIndex=i;sel.dispatchEvent(new Event('change',{bubbles:true}));}paint();};
    const api={wrap,close(){list.hidden=true;wrap.classList.remove('open','up');btn.setAttribute('aria-expanded','false');if(openDD===api)openDD=null;},
      open(){closeAll();list.hidden=false;wrap.classList.add('open');btn.setAttribute('aria-expanded','true');openDD=api;const r=wrap.getBoundingClientRect(),h=Math.min(260,list.scrollHeight)+8,below=window.innerHeight-r.bottom,above=r.top;wrap.classList.toggle('up',below<h&&above>below);setAct(sel.selectedIndex);list.focus({preventScroll:true});}};
    btn.addEventListener('click',()=>{if(list.hidden)api.open();else{api.close();btn.focus();}});
    btn.addEventListener('keydown',e=>{if(['ArrowDown','ArrowUp','Enter',' '].includes(e.key)){e.preventDefault();api.open();}});
    list.addEventListener('click',e=>{const li=e.target.closest('.dd-opt');if(!li)return;choose(+li.dataset.i);api.close();btn.focus();});
    list.addEventListener('mousemove',e=>{const li=e.target.closest('.dd-opt');if(li&&+li.dataset.i!==act)setAct(+li.dataset.i);});
    let typed='',tt=null;
    list.addEventListener('keydown',e=>{
      const k=e.key;
      if(k==='ArrowDown'){e.preventDefault();setAct(act+1);}
      else if(k==='ArrowUp'){e.preventDefault();setAct(act-1);}
      else if(k==='Home'){e.preventDefault();setAct(0);}
      else if(k==='End'){e.preventDefault();setAct(opts.length-1);}
      else if(k==='Enter'||k===' '){e.preventDefault();choose(act);api.close();btn.focus();}
      else if(k==='Escape'){e.preventDefault();api.close();btn.focus();}
      else if(k==='Tab'){api.close();}
      else if(k.length===1&&!e.ctrlKey&&!e.metaKey&&!e.altKey){typed+=k.toLowerCase();clearTimeout(tt);tt=setTimeout(()=>{typed='';},700);const n=opts.findIndex(o=>o.textContent.trim().toLowerCase().startsWith(typed));if(n>=0)setAct(n);}
    });
    sel.addEventListener('change',paint);
    paint();
  });
}
