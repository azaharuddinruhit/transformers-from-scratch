'use strict';
/* ================= site: registry, settings, progress and navigation ================= */
/* The deploy workflow replaces the placeholder with the contents of VERSION; a local copy keeps it. */
const SITE_VERSION='__SITE_VERSION__';
const PARTS=[
  {id:'a',name:'The big picture',goal:'What an LLM computes, how neural networks learn and generate, and why the Transformer replaced earlier networks.',secs:['intro','nn','genai','why']},
  {id:'b',name:'Building the model',goal:'Every layer of a decoder-only Transformer, from text to the probabilities of the next token.',secs:['tokens','position','attention','multihead','block','architecture']},
  {id:'c',name:'Training and using it',goal:'How the model learns, how it generates text, and what it takes to run it.',secs:['training','posttrain','prefs','generation','inference','hardware','quant','flash']},
  {id:'d',name:'The wider picture',goal:'The original encoder–decoder Transformer from the 2017 paper, what today’s models change, and how models read and make images.',secs:['encdec','modern','multimodal','diffusion']}];
/* title, core and optional study minutes, then the level (an index into LEVELS). Minutes: words / 110 + 3 per
   interactive, rounded to 5, code boxes left out. Core leaves out the .opt material and the exact worked examples;
   optional is what they add. Recount when a section changes a lot. */
const SECS={intro:['What is an LLM?',10,0,0],nn:['Neural networks in 15 minutes',15,0,0],genai:['Generative AI: the big picture',15,0,0],why:['Why Transformers?',10,0,0],tokens:['Tokenization and embeddings',15,5,0],position:['Positional information',15,0,1],
  attention:['Self-attention from scratch',15,0,1],multihead:['Multi-head attention',10,0,1],block:['FFN, residuals and normalization',15,5,1],architecture:['Full decoder-only architecture',10,5,1],training:['Training an LLM',15,5,1],generation:['Generation and decoding',10,0,0],
  hardware:['GPUs and CUDA',15,5,1],quant:['Quantization',10,5,1],flash:['FlashAttention',10,5,2],inference:['Inference in practice',10,5,1],posttrain:['From base model to assistant',10,0,1],prefs:['Preference tuning and reasoning',10,5,1],
  encdec:['Encoder–decoder and the 2017 paper',15,10,1],modern:['What modern LLMs change',20,5,2],multimodal:['Multimodal LLMs: reading images',10,5,1],diffusion:['How diffusion models make images',15,0,1]};
const LEVELS=['Foundation','Core','Deep dive'];
const TOPICS={'iv-core':'Transformer fundamentals','iv-rag':'RAG and embeddings','iv-ft':'Fine-tuning and alignment','iv-agents':'Agents and tool use','iv-prod':'Evaluation and production','iv-hw':'Hardware'};
const partOf=id=>PARTS.find(p=>p.secs.includes(id));
const partUrl=p=>'part-'+p.id+'.html';
/* sections that were merged or moved: old links and bookmarks still land in the right place */
const MOVED={scaling:'training',paper:'encdec',openmodels:'models'};
function refInfo(id){
  id=MOVED[id]||id;
  if(SECS[id])return{href:partUrl(partOf(id))+'#'+id,text:SECS[id][0]};
  if(TOPICS[id])return{href:'interview.html#'+id,text:'Interview drill: '+TOPICS[id]};
  const REF={maths:'The maths you need',glossary:'Glossary',notation:'Notation',models:'Open-weight models'};
  if(REF[id])return{href:'reference.html#'+id,text:REF[id]};
  return null;
}
const fmtMin=m=>m>=60?Math.floor(m/60)+' h'+(m%60?' '+m%60+' min':''):m+' min';
/* ---------- progress: one saved record, with a time for every item so two browsers' copies merge ---------- */
/* record: done {id: bool}, quiz {section: [question indexes]}, last (url), drill {id: 'got'|'again'},
   resetAt, and ts, the time each item last changed (a deleted drill mark keeps its time, so the deletion wins too) */
const PROG_KEY='tfs-progress',PROG_MAPS=['done','quiz','drill'];
const progObj=o=>o&&typeof o==='object'&&!Array.isArray(o)?o:{};
function progNorm(raw){
  const d=progObj(raw),t=progObj(d.ts);
  const r={done:{...progObj(d.done)},quiz:{},last:typeof d.last==='string'?d.last:null,drill:{...progObj(d.drill)},resetAt:+d.resetAt||0,
    ts:{last:+t.last||0,done:{...progObj(t.done)},quiz:{...progObj(t.quiz)},drill:{...progObj(t.drill)}}};
  Object.entries(progObj(d.quiz)).forEach(([k,v])=>{if(Array.isArray(v))r.quiz[k]=v.filter(Number.isInteger);});
  /* an item with no time (a record saved before sync existed) counts as older than any change made since */
  if(r.last&&!r.ts.last)r.ts.last=1;
  PROG_MAPS.forEach(k=>Object.keys(r[k]).forEach(id=>{if(!(+r.ts[k][id]>0))r.ts[k][id]=1;}));
  return r;
}
/* the newer copy of each item wins; anything not changed after the latest reset on either side is dropped;
   quiz answers only ever add up, so a section keeps the answers of both copies */
function progMerge(a,b){
  const R=Math.max(a.resetAt,b.resetAt),o={last:null,done:{},quiz:{},drill:{},resetAt:R,ts:{last:0,done:{},quiz:{},drill:{}}};
  const wl=b.ts.last>a.ts.last?b:a;if(wl.ts.last>R){o.last=wl.last;o.ts.last=wl.ts.last;}
  ['done','drill'].forEach(k=>{
    new Set([...Object.keys(a.ts[k]),...Object.keys(b.ts[k])]).forEach(id=>{
      const ta=+a.ts[k][id]||0,tb=+b.ts[k][id]||0,s=tb>ta?b:a,t=Math.max(ta,tb);
      if(t<=R)return;o.ts[k][id]=t;if(id in s[k])o[k][id]=s[k][id];});});
  new Set([...Object.keys(a.quiz),...Object.keys(b.quiz)]).forEach(id=>{
    const keep=[a,b].filter(s=>(+s.ts.quiz[id]||0)>R&&s.quiz[id]);if(!keep.length)return;
    o.quiz[id]=[...new Set([].concat(...keep.map(s=>s.quiz[id])))].sort((x,y)=>x-y);
    o.ts.quiz[id]=Math.max(...keep.map(s=>+s.ts.quiz[id]));});
  return o;
}
/* stamps every item that differs from the last saved copy; true when anything did */
function progStamp(r,old,now){
  let any=false;const st=k=>{r.ts[k]=now;any=true;};
  if(r.last!==old.last)st('last');
  PROG_MAPS.forEach(k=>new Set([...Object.keys(r[k]),...Object.keys(old[k])]).forEach(id=>{
    if(JSON.stringify(r[k][id])!==JSON.stringify(old[k][id])){r.ts[k][id]=now;any=true;}}));
  return any;
}
/* a canonical text of a record, to tell whether a merge changed it */
function progSig(r){
  const c=v=>v&&typeof v==='object'&&!Array.isArray(v)?'{'+Object.keys(v).sort().map(k=>k+':'+c(v[k])).join(',')+'}':JSON.stringify(v);
  return c({last:r.last,done:r.done,quiz:r.quiz,drill:r.drill,resetAt:r.resetAt,ts:r.ts});
}
const onProgress=[];
const store=(()=>{
  let raw=null;try{raw=JSON.parse(localStorage.getItem(PROG_KEY));}catch(e){}
  const d=progNorm(raw);
  try{const old=localStorage.getItem('tfs-done');if(old){JSON.parse(old).forEach(id=>{if(SECS[id]){d.done[id]=true;d.ts.done[id]=1;}});localStorage.removeItem('tfs-done');}}catch(e){}
  let snap=JSON.parse(JSON.stringify(d));
  const write=()=>{try{localStorage.setItem(PROG_KEY,JSON.stringify(d));}catch(e){}snap=JSON.parse(JSON.stringify(d));};
  /* folds in what another tab saved and, from a sync, the gist's copy; d stays the same object, so P keeps working */
  function absorb(remote){
    progStamp(d,snap,Date.now());
    let m=d,cur=null;try{cur=JSON.parse(localStorage.getItem(PROG_KEY));}catch(e){}
    if(cur)m=progMerge(m,progNorm(cur));if(remote)m=progMerge(m,remote);
    const before=progSig(d);Object.assign(d,m);write();
    if(progSig(d)!==before)onProgress.forEach(f=>f());
  }
  return{d,absorb,
    save(){if(progStamp(d,snap,Date.now())){absorb(null);syncSoon();}else write();},
    reset(){const now=Date.now();Object.assign(d,{done:{},quiz:{},drill:{},last:null,resetAt:now,ts:{last:0,done:{},quiz:{},drill:{}}});write();}};
})();
const P=store.d;
window.addEventListener('storage',e=>{if(e.key===PROG_KEY)store.absorb(null);});

/* ---------- sync through a secret GitHub gist ----------
   localStorage stays the working copy; the gist holds a copy every browser can reach, merged item by item (progMerge).
   The token is pasted once per browser in Settings and kept only in that browser: never write it into a file. */
const GIST_ID='393397f29e536c680c5b06da85df9093',GIST_FILE='tfs-progress.json',GIST_API='https://api.github.com/gists/'+GIST_ID;
const TOKEN_KEY='tfs-gist-token',SYNC_AT_KEY='tfs-sync-at',SYNC_DELAY=2000;
const sync={timer:null,running:null,again:false,busy:false,err:''};
const syncToken=()=>{try{return localStorage.getItem(TOKEN_KEY)||'';}catch(e){return '';}};
const syncAt=()=>{try{return +localStorage.getItem(SYNC_AT_KEY)||0;}catch(e){return 0;}};
function syncSoon(){if(!syncToken())return;clearTimeout(sync.timer);sync.timer=setTimeout(()=>syncNow(),SYNC_DELAY);}
async function gistRead(token){
  const res=await fetch(GIST_API,{cache:'no-store',headers:{Authorization:'Bearer '+token,Accept:'application/vnd.github+json'}});
  if(res.status===401)throw new Error('GitHub rejected the token');
  if(res.status===403||res.status===404)throw new Error('the gist was not found, or the token cannot reach it (it needs the gist scope)');
  if(!res.ok)throw new Error('GitHub answered HTTP '+res.status);
  const file=(await res.json()).files[GIST_FILE];
  if(!file)return null;   // a fresh gist: the first sync adds the file
  let text=file.content;
  if(file.truncated)text=await (await fetch(file.raw_url,{cache:'no-store'})).text();
  if(!text.trim())return null;
  try{return progNorm(JSON.parse(text));}catch(e){throw new Error(GIST_FILE+' in the gist is not valid JSON');}
}
async function gistWrite(token,rec,keepalive){
  const body=JSON.stringify({files:{[GIST_FILE]:{content:JSON.stringify(rec,null,1)}}});
  const res=await fetch(GIST_API,{method:'PATCH',keepalive:!!keepalive&&body.length<60000,headers:{Authorization:'Bearer '+token,Accept:'application/vnd.github+json'},body});
  if(!res.ok)throw new Error('GitHub answered HTTP '+res.status+' while saving');
}
/* reads the gist, merges it in, and writes the merge back when the gist lacks something; never two at once */
function syncNow(keepalive){
  const token=syncToken();if(!token)return Promise.resolve();
  clearTimeout(sync.timer);sync.timer=null;
  if(sync.running){sync.again=true;return sync.running;}
  sync.busy=true;paintSync();
  sync.running=(async()=>{
    try{
      const remote=await gistRead(token);
      store.absorb(remote);
      if(!remote||progSig(remote)!==progSig(P))await gistWrite(token,P,keepalive);
      sync.err='';try{localStorage.setItem(SYNC_AT_KEY,String(Date.now()));}catch(e){}
    }catch(e){sync.err=e&&e.message||String(e);}
    finally{sync.running=null;sync.busy=false;paintSync();if(sync.again){sync.again=false;syncNow();}}
  })();
  return sync.running;
}
document.addEventListener('visibilitychange',()=>{
  if(!syncToken())return;
  if(document.visibilityState==='hidden'){if(sync.timer)syncNow(true);}else syncNow();
});
const agoText=t=>{const s=Math.round((Date.now()-t)/1000);return s<60?'just now':s<3600?Math.round(s/60)+' min ago':s<86400?Math.round(s/3600)+' h ago':new Date(t).toLocaleDateString();};
function paintSync(){
  const st=$('#syncStat');if(!st)return;const on=!!syncToken(),bad=on&&!sync.busy&&!!sync.err;
  const txt=!on?'Off':sync.busy?'Syncing…':bad?'Failed':syncAt()?'Synced '+agoText(syncAt()):'On';
  st.textContent=txt;
  $('#syncVal').innerHTML=(on?'<i class="sdot'+(bad?' bad':'')+'" aria-hidden="true"></i>':'')+txt;
  const note=$('#syncNote');note.classList.toggle('bad',bad);
  note.textContent=!on?'Keeps your progress and path in a secret GitHub gist, so they follow you between browsers.':bad?'The last sync failed: '+sync.err+'.':'Syncs when a page opens, two seconds after a change, and when you leave or return to the tab.';
  const dot=$('#syncStat').previousElementSibling;dot.hidden=!on;dot.classList.toggle('bad',bad);
  $('#syncOn').hidden=on;$('#syncNowBtn').hidden=$('#syncOff').hidden=!on;$('#syncNowBtn').disabled=sync.busy;
}
function initSync(closeSet){
  if(!$('#syncOn'))return;
  const dlg=document.createElement('dialog');dlg.className='dlg';dlg.setAttribute('aria-labelledby','syTitle');
  dlg.innerHTML='<form method="dialog"><h2 id="syTitle">Sync progress with GitHub</h2>'
    +'<p>Paste a GitHub token that can write gists: a classic token with the <code>gist</code> scope, or a fine-grained token with Gists set to read and write. <a href="https://github.com/settings/tokens/new?scopes=gist&amp;description=Transformers%20guide%20sync" target="_blank" rel="noopener">Create a classic token</a>.</p>'
    +'<p>The token is kept only in this browser and sent only to GitHub. Your progress here is merged with the gist’s copy, so nothing is lost on either side.</p>'
    +'<label class="ctl" style="max-width:none">Token<input type="password" id="syTok" autocomplete="off" spellcheck="false"></label>'
    +'<p class="syerr" id="syErr" role="alert" hidden></p>'
    +'<div class="acts"><button type="button" class="btn" id="syNo">Cancel</button><button class="btn pri" id="syGo" value="connect">Connect</button></div></form>';
  const off=document.createElement('dialog');off.className='dlg';off.setAttribute('aria-labelledby','soTitle');
  off.innerHTML='<form method="dialog"><h2 id="soTitle">Disconnect sync?</h2><p>This browser stops syncing and forgets the token. Progress saved here and in the gist is kept. To connect again you will need a token.</p>'
    +'<div class="acts"><button type="button" class="btn" id="soNo" autofocus>Cancel</button><button type="button" class="btn" id="soYes">Disconnect</button></div></form>';
  document.body.append(dlg,off);
  [dlg,off].forEach(x=>x.addEventListener('click',e=>{if(e.target===x)x.close('cancel');}));
  const err=$('#syErr',dlg),inp=$('#syTok',dlg),go=$('#syGo',dlg);
  $('#syNo',dlg).addEventListener('click',()=>dlg.close('cancel'));
  $('#syncOn').addEventListener('click',()=>{closeSet(false);inp.value='';err.hidden=true;dlg.returnValue='';dlg.showModal();inp.focus();});
  $('form',dlg).addEventListener('submit',async e=>{
    e.preventDefault();const tok=inp.value.trim();
    if(!tok){err.textContent='Paste a token first.';err.hidden=false;inp.focus();return;}
    go.disabled=true;go.textContent='Checking…';err.hidden=true;
    try{await gistRead(tok);}catch(x){err.textContent='Could not connect: '+x.message+'.';err.hidden=false;go.disabled=false;go.textContent='Connect';inp.focus();return;}
    go.disabled=false;go.textContent='Connect';
    try{localStorage.setItem(TOKEN_KEY,tok);}catch(x){err.textContent='This browser does not allow saving the token.';err.hidden=false;return;}
    dlg.close('connected');paintSync();syncNow();
  });
  dlg.addEventListener('close',()=>{if(dlg.returnValue!=='connected')$('#setBtn').focus();});
  $('#syncOff').addEventListener('click',()=>{closeSet(false);off.returnValue='';off.showModal();});
  $('#soNo',off).addEventListener('click',()=>off.close());
  $('#soYes',off).addEventListener('click',()=>{try{localStorage.removeItem(TOKEN_KEY);localStorage.removeItem(SYNC_AT_KEY);}catch(e){}clearTimeout(sync.timer);sync.err='';paintSync();off.close();});
  off.addEventListener('close',()=>$('#setBtn').focus());
  $('#syncNowBtn').addEventListener('click',()=>syncNow());
  paintSync();
  syncNow();
}
const minsOf=id=>SECS[id][1];
const partStats=p=>{const n=p.secs.filter(id=>P.done[id]).length;return{n,total:p.secs.length,left:p.secs.filter(id=>!P.done[id]).reduce((a,id)=>a+minsOf(id),0)};};
/* the study time of a section or of the whole guide, as HTML: core minutes, then an "Optional"-style pill with
   what the optional material adds */
const timeText=(core,opt)=>'about '+fmtMin(core)+(opt?' <span class="optpill">+'+fmtMin(opt)+' optional</span>':'');
/* optional material (.opt) gets a small "Optional" pill on each optional heading and box, and at the start of
   each run of plain optional text; the Python boxes say what they are in their titles, so they are left unmarked */
function initOptional(){
  const isOpt=e=>e&&e.classList.contains('opt')&&!e.matches('details.code');
  $$('.opt').forEach(e=>{
    const plain=e.matches('p,ul,ol,tr,.tex')||(e.matches('div')&&!e.matches('.widget,.worked,.callout'));
    if(!isOpt(e)||(plain&&isOpt(e.previousElementSibling)))return;
    const pill=document.createElement('span');pill.className='optpill';pill.textContent='Optional';
    /* a real space beside the pill, so copied text and screen readers get "misconceptions Optional" and a narrow
       screen wraps the pill at the space */
    const after=t=>t.append(' ',pill);
    if(e.matches('.try'))after($('b',e));
    else if(e.matches('details'))after($('summary',e));
    else if(e.matches('.widget,.worked'))after($('h4',e));
    else if(e.matches('tr'))after(e.cells[0]);
    else if(e.matches('h3'))after(e);
    else if(!e.matches('ul,ol')){pill.classList.add('lead');e.prepend(pill,' ');}
  });
}
/* the Settings menu: an iOS-style grouped list built here once for every page, with sub-pages for the path, progress and
   sync; a popover under its button, a bottom sheet on phones */
const ICONS={theme:'<path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z"/>',
  prog:'<circle cx="12" cy="12" r="8"/><path d="M8.5 12.5l2.5 2.5 4.5-5"/>',sync:'<path d="M7 18a4 4 0 0 1-.6-8A6 6 0 0 1 18 9a4.5 4.5 0 0 1-.5 9z"/>',
  reset:'<path d="M4 12a8 8 0 1 0 2.4-5.7M4 4v5h5"/>',chev:'<path d="M9 6l6 6-6 6"/>',back:'<path d="M15 6l-6 6 6 6"/>'};
const svgI=k=>'<svg viewBox="0 0 24 24" aria-hidden="true">'+ICONS[k]+'</svg>';
const sIc=k=>'<span class="sic ic-'+k+'">'+svgI(k)+'</span>';
const sNav=(to,k,label,valId)=>'<button type="button" class="sitem" data-go="'+to+'">'+sIc(k)+'<span class="slbl">'+label+'</span><span class="sval" id="'+valId+'"></span><span class="schev">'+svgI('chev')+'</span></button>';
const sSwitch=(id,k,label)=>'<button type="button" class="sitem" role="switch" aria-checked="false" id="'+id+'">'+sIc(k)+'<span class="slbl">'+label+'</span><span class="swt" aria-hidden="true"></span></button>';
const SET_PAGES={main:'Settings',prog:'Progress',sync:'GitHub sync'};
const SET_MENU='<div class="smhead"><button type="button" class="sbtn" id="setBack" aria-label="Back to Settings" hidden>'+svgI('back')+'</button><span id="setTitle">Settings</span></div>'
  +'<div class="spages">'
  +'<div class="spage" data-page="main">'
  +'<div class="sgroup">'+sSwitch('thSw','theme','Dark theme')+'</div>'
  +'<div class="sgroup">'+sNav('prog','prog','Progress','progVal')+sNav('sync','sync','GitHub sync','syncVal')+'</div>'
  +'<div class="sgroup"><button type="button" class="sitem danger" id="pgReset">'+sIc('reset')+'<span class="slbl">Reset progress</span></button></div>'
  +'<p class="sver" id="setVer"></p></div>'
  +'<div class="spage" data-page="prog"><div class="sgroup" id="progList"></div><div class="sgroup" id="progDrill"></div><p class="sfoot">A section is done when every quiz question has been answered correctly once, or when you press its Done button.</p></div>'
  +'<div class="spage" data-page="sync"><div class="sgroup"><div class="sitem sinfo"><span class="slbl">Status</span><span class="sval"><i class="sdot" aria-hidden="true" hidden></i><span id="syncStat" aria-live="polite"></span></span></div></div>'
  +'<p class="sfoot" id="syncNote"></p>'
  +'<div class="sgroup"><button type="button" class="sitem saccent" id="syncOn"><span class="slbl">Connect a gist token…</span></button><button type="button" class="sitem saccent" id="syncNowBtn"><span class="slbl">Sync now</span></button><button type="button" class="sitem danger" id="syncOff"><span class="slbl">Disconnect</span></button></div>'
  +'<p class="sfoot">The token is kept only in this browser and sent only to GitHub. The theme stays per browser.</p></div>'
  +'</div>';
function paintProgSum(){
  const v=$('#progVal');if(!v)return;
  const ids=Object.keys(SECS),n=ids.filter(id=>P.done[id]).length;
  v.textContent=n+' of '+ids.length;
  $('#progList').innerHTML=PARTS.map(p=>{const st=partStats(p);
    return '<div class="sitem sinfo"><span class="spl'+(st.n===st.total?' full':'')+'" aria-hidden="true">'+p.id.toUpperCase()+'</span><span class="slbl">'+p.name+'<span class="sbar"><i style="width:'+(100*st.n/st.total)+'%"></i></span></span><span class="sval">'+st.n+' of '+st.total+'</span></div>';}).join('');
  const marks=Object.values(P.drill),got=marks.filter(m=>m==='got').length,again=marks.filter(m=>m==='again').length;
  $('#progDrill').innerHTML='<a class="sitem" href="interview.html"><span class="slbl">Interview drill</span><span class="sval">'+(got||again?got+' got it · '+again+' to review':'Not started')+'</span><span class="schev">'+svgI('chev')+'</span></a>';
}
function initSettings(){
  const setBtn=$('#setBtn'),menu=$('#setMenu');
  menu.innerHTML=SET_MENU;
  $('#setVer').textContent=SITE_VERSION.startsWith('__')?'Version: local copy':'Version '+SITE_VERSION;
  /* the header's backdrop blur would trap a fixed element, so the menu lives in <body> */
  const scrim=document.createElement('div');scrim.className='setscrim';scrim.hidden=true;
  document.body.append(scrim,menu);
  const root=document.documentElement,thSw=$('#thSw');
  // Light or dark; until the reader chooses, the page follows the system setting
  const sysTheme=()=>window.matchMedia&&matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light';
  const curTheme=()=>root.getAttribute('data-theme')||sysTheme();
  const markTheme=()=>thSw.setAttribute('aria-checked',curTheme()==='dark');
  try{const t=localStorage.getItem('tfs-theme');if(t==='light'||t==='dark')root.setAttribute('data-theme',t);}catch(e){}
  markTheme();
  thSw.addEventListener('click',()=>{const t=curTheme()==='dark'?'light':'dark';root.setAttribute('data-theme',t);try{localStorage.setItem('tfs-theme',t);}catch(e){}markTheme();});
  /* sub-pages slide in from the right; the box animates to the new page's height */
  const pages=$('.spages',menu),back=$('#setBack'),title=$('#setTitle');let cur='main',from=null;
  const go=(to,anim)=>{
    const h0=pages.offsetHeight;
    $$('.spage',menu).forEach(p=>{p.classList.toggle('on',p.dataset.page===to);p.classList.toggle('left',to!=='main'&&p.dataset.page==='main');});
    if(to!=='main')from=cur==='main'?$('[data-go="'+to+'"]',menu):from;
    cur=to;back.hidden=to==='main';title.textContent=SET_PAGES[to];
    const h1=$('.spage.on',menu).offsetHeight;
    if(anim&&h0!==h1){pages.style.height=h0+'px';pages.offsetHeight;pages.style.height=h1+'px';setTimeout(()=>{pages.style.height='';},260);}
    const f=to==='main'?from:back;if(anim&&f)f.focus();
  };
  $$('[data-go]',menu).forEach(b=>b.addEventListener('click',()=>go(b.dataset.go,true)));
  back.addEventListener('click',()=>go('main',true));
  go('main',false);
  const phone=()=>window.matchMedia('(max-width:600px)').matches;
  const place=()=>{if(phone()){menu.style.top=menu.style.right='';return;}const r=setBtn.getBoundingClientRect();menu.style.top=(r.bottom+10)+'px';menu.style.right=(document.documentElement.clientWidth-r.right)+'px';};
  const closeSet=ret=>{if(menu.hidden)return;menu.hidden=true;scrim.hidden=true;setBtn.setAttribute('aria-expanded','false');if(ret)setBtn.focus();};
  const openSet=()=>{go('main',false);paintProgSum();paintSync();markTheme();place();menu.hidden=false;scrim.hidden=false;setBtn.setAttribute('aria-expanded','true');$('.sitem',menu).focus();};
  /* reset asks first in a modal <dialog>: it traps focus, closes on Escape and returns "reset" only from its button */
  const dlg=document.createElement('dialog');dlg.className='dlg';dlg.setAttribute('aria-labelledby','rsTitle');
  dlg.innerHTML='<form method="dialog"><h2 id="rsTitle">Reset progress?</h2><p>This clears the finished sections, quiz answers and drill marks saved in this browser and, when sync is on, in every synced browser. Your path and theme stay. You cannot undo this.</p>'
    +'<div class="acts"><button class="btn" value="cancel" autofocus>Cancel</button><button class="btn danger" value="reset">Reset progress</button></div></form>';
  document.body.append(dlg);
  dlg.addEventListener('click',e=>{if(e.target===dlg)dlg.close('cancel');});
  dlg.addEventListener('close',()=>{if(dlg.returnValue==='reset'){store.reset();location.reload();}else setBtn.focus();});
  $('#pgReset').addEventListener('click',()=>{closeSet(false);dlg.returnValue='';dlg.showModal();});
  setBtn.addEventListener('click',()=>{if(menu.hidden)openSet();else closeSet(true);});
  const inside=el=>el&&(menu.contains(el)||setBtn.contains(el));
  document.addEventListener('click',e=>{if(!menu.hidden&&e.target.isConnected&&!inside(e.target))closeSet(false);});
  document.addEventListener('keydown',e=>{if(e.key!=='Escape'||menu.hidden)return;if(cur!=='main')go('main',true);else closeSet(true);});
  menu.addEventListener('focusout',e=>{if(e.relatedTarget&&!inside(e.relatedTarget))closeSet(false);});
  onProgress.push(paintProgSum);
  let rt=null,lastW=window.innerWidth;window.addEventListener('resize',()=>{if(!menu.hidden)place();if(window.innerWidth===lastW)return;lastW=window.innerWidth;clearTimeout(rt);rt=setTimeout(rerender,150);});
  initSync(closeSet);
  const bar=$('#progBar');window.addEventListener('scroll',()=>{const h=document.documentElement;bar.style.width=(100*Math.max(0,Math.min(1,h.scrollTop/Math.max(1,h.scrollHeight-h.clientHeight)))).toFixed(1)+'%';},{passive:true});
}
/* cross-references show the target's name and link across pages */
function initXrefs(){
  $$('a.xref').forEach(a=>{const r=refInfo(a.dataset.ref);if(!r)return;const here=r.href.split('#')[0]===location.pathname.split('/').pop();a.href=here?'#'+a.dataset.ref:r.href;a.textContent=r.text;});
}
/* quizzes: a section is done once every question has been answered correctly (or when marked by hand) */
function initQuiz(onChange){
  document.addEventListener('click',e=>{
    const b=e.target.closest('.qo');if(!b)return;
    const f=b.closest('fieldset'),sec=b.closest('section.ch'),x=$('.qx',f),ok=!!b.dataset.ok;
    $$('.qo',f).forEach(o=>{o.classList.remove('wrong');if(o.dataset.ok)o.classList.add('right');});
    if(!ok)b.classList.add('wrong');x.firstChild.textContent=ok?'Correct.':'Not quite.';x.hidden=false;
    if(!ok||!sec)return;
    const qs=$$('.quiz fieldset',sec),i=qs.indexOf(f),got=P.quiz[sec.id]||(P.quiz[sec.id]=[]);
    if(!got.includes(i))got.push(i);
    if(got.length>=qs.length&&P.done[sec.id]===undefined)P.done[sec.id]=true;
    store.save();onChange(sec.id);
  });
}
const quizScore=sec=>{const n=$$('.quiz fieldset',sec).length;return n?{got:Math.min(n,(P.quiz[sec.id]||[]).length),n}:null;};
/* ---------------- part pages: one view at a time ---------------- */
function initPart(){
  const part=PARTS.find(p=>'part-'+p.id===document.body.dataset.page);if(!part)return;
  const L=part.id.toUpperCase(),next=PARTS[PARTS.indexOf(part)+1];
  const views=$$('#main>section'),ids=views.map(v=>v.id),secs=views.filter(v=>v.classList.contains('ch'));
  const label=id=>id==='start'?'Overview':id==='complete'?'Part complete':SECS[id][0];
  $('#partGoal').textContent=part.goal;

  /* meta line under each title, and one footer per section */
  secs.forEach((s,i)=>{
    const m=document.createElement('p');m.className='rt';s.querySelector('h2').insertAdjacentElement('afterend',m);
    const prev=ids[ids.indexOf(s.id)-1],nx=ids[ids.indexOf(s.id)+1],bridge=$('.bridge',s);
    const nav=document.createElement('nav');nav.className='snav';nav.setAttribute('aria-label','Section navigation');
    nav.innerHTML='<div class="snrow"><a class="prev" href="#'+prev+'">← '+(prev==='start'?'Part overview':'Previous')+'</a>'
      +'<button type="button" class="doneBtn" aria-pressed="false"></button></div>'
      +'<a class="nextBtn" href="#'+nx+'"><small>Next</small><span>'+(nx==='complete'?'Finish Part '+L:label(nx))+' →</span></a>';
    if(bridge)nav.insertBefore(bridge,$('.nextBtn',nav));
    s.appendChild(nav);
    $('.doneBtn',nav).addEventListener('click',()=>{P.done[s.id]=!P.done[s.id];store.save();paint();});
  });

  const toc=$('#toc'),scrim=$('#scrim'),mMid=$('#mMid'),mbar=$('#mbar');
  let cur=null;
  function paint(){
    const st=partStats(part),pos=secs.findIndex(s=>s.id===cur);
    secs.forEach((s,i)=>{const on=!!P.done[s.id],q=quizScore(s);
      $('.rt',s).innerHTML='Part '+L+' · '+(i+1)+' of '+secs.length+' · '+timeText(minsOf(s.id),SECS[s.id][2])+' <span class="lvl lv'+SECS[s.id][3]+'">'+LEVELS[SECS[s.id][3]]+'</span>';
      const b=$('.doneBtn',s);b.setAttribute('aria-pressed',on);b.classList.toggle('on',on);b.textContent=on?'✓ Done':'Mark as done';
      const qs=$('.qscore',s);if(qs&&q)qs.textContent=q.got?q.got+' of '+q.n+' correct':'';});
    toc.innerHTML='<div class="tochead"><a href="index.html" class="tochome">← All parts</a><button type="button" class="btn tiny" id="tocClose">Close</button></div>'
      +'<p class="tocpart">Part '+L+'<b>'+part.name+'</b></p>'
      +'<div class="prog"><div class="pline"><b>'+st.n+' of '+st.total+' done</b><span>'+(st.left?'about '+fmtMin(st.left)+' left':'complete')+'</span></div><span class="pbar"><i style="width:'+(100*st.n/st.total)+'%"></i></span></div>'
      +'<ol>'+ids.map(id=>{const s=SECS[id],done=s&&P.done[id];return '<li class="'+(id===cur?'on ':'')+(done?'done':'')+'"><a href="#'+id+'"'+(id===cur?' aria-current="page"':'')+'><span>'+label(id)+(done?'<span class="vh">, done</span>':'')+'</span>'+(s?'<small>'+minsOf(id)+' min</small>':'')+'</a></li>';}).join('')+'</ol>';
    $('#tocClose').addEventListener('click',()=>closeSheet(true));
    $('#mNum').textContent=pos<0?'':(pos+1)+'/'+secs.length;$('#mTxt').textContent=label(cur);
    mMid.setAttribute('aria-label',(pos<0?label(cur):'Section '+(pos+1)+' of '+secs.length+', '+label(cur))+'. '+st.n+' of '+st.total+' done. Open contents');
    $('#mFill').style.width=(100*st.n/st.total)+'%';
    $('#mPrev').disabled=ids.indexOf(cur)<=0;$('#mNext').disabled=ids.indexOf(cur)>=ids.length-1;
    $('#crumb').innerHTML='<a href="index.html">Home</a><span aria-hidden="true">/</span>Part '+L+(pos<0?'':'<span class="steps">'+(pos+1)+' of '+secs.length+'</span>');
    if(cur==='start')paintStart();if(cur==='complete')paintComplete();
  }
  function paintStart(){
    $('#partList').innerHTML=secs.map(s=>'<li class="'+(P.done[s.id]?'done':'')+'"><a href="#'+s.id+'">'+SECS[s.id][0]+'</a><small>'+minsOf(s.id)+' min</small></li>').join('');
    const first=secs.find(s=>!P.done[s.id]);
    $('#partGo').innerHTML=first?'<a class="btn pri big" href="#'+first.id+'">'+(first===secs[0]?'Start':'Continue')+': '+SECS[first.id][0]+' →</a>':'<a class="btn big" href="#complete">Review the key takeaways →</a>';
  }
  function paintComplete(){
    const st=partStats(part);let got=0,n=0;secs.forEach(s=>{const q=quizScore(s);if(q){got+=q.got;n+=q.n;}});
    $('#partDone').innerHTML='<p class="kicker">Part '+L+'</p><h1>'+(st.n===st.total?'Part '+L+' complete':st.n+' of '+st.total+' sections done')+'</h1>'
      +'<p class="lede">'+(n?'You have answered '+got+' of '+n+' check questions correctly.':'')+(st.n<st.total?' Sections not yet done: '+secs.filter(s=>!P.done[s.id]).map(s=>'<a href="#'+s.id+'">'+SECS[s.id][0]+'</a>').join(', ')+'.':'')+'</p>'
      +'<p class="acts">'+(next?'<a class="btn pri big" href="'+partUrl(next)+'">Start Part '+next.id.toUpperCase()+': '+next.name+' →</a>':'<a class="btn pri big" href="interview.html">Go to the interview drill →</a>')+' <a class="btn big" href="index.html">All parts</a></p>'
      +'<h2>Key takeaways</h2>'+secs.map(s=>'<h3>'+SECS[s.id][0]+'</h3>'+$('ul.take',s).outerHTML).join('');
    renderTex($('#partDone'));
  }
  function show(id,anchor){
    if(!ids.includes(id))id='start';
    const changed=id!==cur;cur=id;
    views.forEach(v=>{v.hidden=v.id!==id;});
    if(SECS[id]){P.last=partUrl(part)+'#'+id;store.save();}
    document.title=label(id)+' · Part '+L+' · Transformers and LLMs from Scratch';
    paint();
    if(changed)rerender();
    const t=anchor&&document.getElementById(anchor);
    if(t&&t!==$('#'+id))t.scrollIntoView();else window.scrollTo(0,0);
  }
  const route=()=>{const h=decodeURIComponent(location.hash.slice(1));if(MOVED[h]){location.replace(refInfo(h).href);return;}const el=h&&document.getElementById(h),v=el&&el.closest('#main>section');show(v?v.id:h,h);};
  window.addEventListener('hashchange',route);
  initQuiz(()=>paint());
  onProgress.push(paint);

  /* contents: side list on wide screens, bottom sheet behind the pill on narrow ones */
  const openSheet=()=>{mbar.classList.remove('hid');toc.classList.add('open');scrim.hidden=false;mMid.setAttribute('aria-expanded','true');const t=$('li.on a',toc)||$('a',toc);if(t)t.focus({preventScroll:true});};
  function closeSheet(ret){if(!toc.classList.contains('open'))return;toc.classList.remove('open');scrim.hidden=true;mMid.setAttribute('aria-expanded','false');if(ret)mMid.focus();}
  mMid.addEventListener('click',()=>toc.classList.contains('open')?closeSheet(true):openSheet());
  scrim.addEventListener('click',()=>closeSheet(true));
  toc.addEventListener('click',e=>{if(e.target.closest('ol a'))closeSheet(false);});
  document.addEventListener('keydown',e=>{if(e.key==='Escape')closeSheet(true);});
  const step=d=>{const i=ids.indexOf(cur)+d;if(i>=0&&i<ids.length)location.hash='#'+ids[i];};
  $('#mPrev').addEventListener('click',()=>step(-1));$('#mNext').addEventListener('click',()=>step(1));
  /* the pill hides while scrolling down and returns on scroll up, near the top or bottom, and at a section's footer */
  let lastY=window.scrollY,navIn=0;
  if('IntersectionObserver' in window){const io=new IntersectionObserver(es=>{es.forEach(e=>{navIn+=e.isIntersecting?1:(e.target._in?-1:0);e.target._in=e.isIntersecting;});navIn=Math.max(0,navIn);if(navIn)mbar.classList.remove('hid');});$$('.snav').forEach(n=>io.observe(n));}
  window.addEventListener('scroll',()=>{const y=window.scrollY,dy=y-lastY;if(Math.abs(dy)<8)return;
    const keep=y<120||window.innerHeight+y>=document.documentElement.scrollHeight-120||navIn>0||toc.classList.contains('open')||mbar.contains(document.activeElement);
    mbar.classList.toggle('hid',dy>0&&!keep);lastY=y;},{passive:true});
  mbar.addEventListener('focusin',()=>mbar.classList.remove('hid'));
  route();
}
/* ---------------- home: path choice and the journey ---------------- */
function initHome(){
  if(document.body.dataset.page!=='home')return;
  /* links to the old single page (#attention, #notation, …) */
  const h=location.hash.slice(1),r=h&&refInfo(h);if(r){location.replace(r.href);return;}
  function paint(){
    const sum=i=>Object.keys(SECS).reduce((a,id)=>a+SECS[id][i],0);
    $('#studyTime').innerHTML='Study time: '+timeText(sum(1),sum(2));
    const lastId=P.last&&P.last.split('#')[1];
    $('#resume').innerHTML=lastId&&SECS[lastId]?'<a class="btn pri big" href="'+P.last+'">Continue: '+SECS[lastId][0]+' →</a>':'<a class="btn pri big" href="part-a.html">Start with Part A →</a>';
    $('#journey').innerHTML=PARTS.map(p=>{const st=partStats(p),state=st.n===st.total?'complete':st.n?'started':'';
      return '<li class="'+state+'"><a class="jpart" href="'+partUrl(p)+'"><span class="jl">'+p.id.toUpperCase()+'</span><span class="jt"><b>'+p.name+'</b><span>'+p.goal+'</span>'
        +'<small>'+st.total+' sections · '+(state==='complete'?'complete':st.n?st.n+' done, about '+fmtMin(st.left)+' left':'about '+fmtMin(st.left))+'</small></span></a>'
        +'<ol class="jsecs">'+p.secs.map(id=>'<li class="'+(P.done[id]?'done':'')+'"><a href="'+partUrl(p)+'#'+id+'">'+SECS[id][0]+(P.done[id]?'<span class="vh">, done</span>':'')+'</a></li>').join('')+'</ol></li>';}).join('');
    const got=Object.values(P.drill).filter(v=>v==='got').length;
    if(got)$('#drillStat').textContent=got+' question'+(got>1?'s':'')+' marked “Got it” so far';
  }
  onProgress.push(paint);paint();
}
/* ---------------- interview drill ---------------- */
function initDrill(){
  if(document.body.dataset.page!=='interview')return;
  const qas=$$('.qa'),f={topic:'all',status:'all'};
  qas.forEach(qa=>{
    const sec=qa.closest('section.topic');qa.dataset.topic=sec.id;qa.dataset.id=sec.id+'-'+$('.qn',qa).textContent;
    const ans=document.createElement('div');ans.className='ans';ans.hidden=true;ans.id='ans-'+qa.dataset.id;
    while(qa.children.length>1)ans.appendChild(qa.children[1]);
    const act=document.createElement('div');act.className='qact';
    act.innerHTML='<button type="button" class="btn tiny show" aria-expanded="false" aria-controls="'+ans.id+'">Show answer</button><span class="mk"><button type="button" class="btn tiny" data-m="got" aria-pressed="false">Got it</button><button type="button" class="btn tiny" data-m="again" aria-pressed="false">Review again</button></span>';
    qa.append(act,ans);
    $('.show',act).addEventListener('click',e=>{const open=ans.hidden;ans.hidden=!open;e.target.setAttribute('aria-expanded',open);e.target.textContent=open?'Hide answer':'Show answer';});
    $$('[data-m]',act).forEach(b=>b.addEventListener('click',()=>{const m=b.dataset.m;if(P.drill[qa.dataset.id]===m)delete P.drill[qa.dataset.id];else P.drill[qa.dataset.id]=m;store.save();paint();}));
  });
  /* one slim pinned row: two dropdowns (turned into the site's dropdowns by initSelects) and the count */
  const pick=(k,label,opts)=>'<div class="dsel"><select data-k="'+k+'" aria-label="'+label+'">'+opts.map(([v,t])=>'<option value="'+v+'">'+t+'</option>').join('')+'</select></div>';
  $('#drillbar').innerHTML=pick('topic','Topic',[['all','All topics']].concat(Object.entries(TOPICS)))
    +pick('status','Show',[['all','All questions'],['new','Not marked'],['again','Review again'],['got','Got it']])
    +'<p class="dcount" aria-live="polite"><span class="dc-got"></span><span class="dc-more"></span></p>';
  $('#drillbar').addEventListener('change',e=>{f[e.target.dataset.k]=e.target.value;paint();});
  function paint(){
    let got=0,again=0;
    qas.forEach(qa=>{const m=P.drill[qa.dataset.id];if(m==='got')got++;if(m==='again')again++;
      $$('[data-m]',qa).forEach(b=>b.setAttribute('aria-pressed',b.dataset.m===m));
      qa.classList.toggle('got',m==='got');
      qa.hidden=!((f.topic==='all'||qa.dataset.topic===f.topic)&&(f.status==='all'||(f.status==='new'?!m:m===f.status)));});
    /* hide headings, and whole topics, whose questions are all filtered out */
    $$('section.topic').forEach(sec=>{let any=false,h=null,hAny=false;
      const end=()=>{if(h)h.hidden=!hAny;};
      Array.from(sec.children).forEach(c=>{if(c.tagName==='H3'){end();h=c;hAny=false;}else if(c.classList.contains('qa')&&!c.hidden){any=true;hAny=true;}});
      end();sec.hidden=!any;});
    const shown=qas.filter(q=>!q.hidden).length;
    /* the second part is hidden on phones, where the row is narrow */
    $('.dc-got').textContent=got+'/'+qas.length+' got it';
    $('.dc-more').textContent=' · '+again+' to review'+(shown<qas.length?' · showing '+shown:'');
  }
  onProgress.push(paint);paint();
  const h=location.hash.slice(1);if(h&&document.getElementById(h))document.getElementById(h).scrollIntoView();
}

[initSettings,initOptional,initCode,...PAGE_INITS,initXrefs,initPart,initHome,initDrill,initSelects].forEach(f=>{try{f();}catch(e){console.error(f.name,e);}});
renderTex(document);
