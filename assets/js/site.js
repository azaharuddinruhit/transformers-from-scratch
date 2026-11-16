'use strict';
/* ================= site: registry, settings, progress and navigation ================= */
/* The deploy workflow replaces the placeholder with the contents of VERSION; a local copy keeps it. */
const SITE_VERSION='__SITE_VERSION__';
const PARTS=[
  {id:'a',name:'The big picture',goal:'What an LLM computes, and why the Transformer replaced recurrent networks.',secs:['intro','nn','why']},
  {id:'b',name:'Building the model',goal:'Every layer of a decoder-only Transformer, from text to the probabilities of the next token.',secs:['tokens','position','attention','multihead','block','architecture']},
  {id:'c',name:'Training and using it',goal:'How the model learns, how it generates text, and what it takes to run it.',secs:['training','posttrain','prefs','generation','inference','hardware']},
  {id:'d',name:'The wider picture',goal:'The original encoder–decoder Transformer from the 2017 paper, and what today’s models change.',secs:['encdec','modern']}];
/* title, study minutes for Full depth and Quick tour, then the level (an index into LEVELS).
   Minutes: words / 110 + 3 per interactive, rounded to 5, code boxes left out (Full depth drops .s-only; Quick tour
   drops .opt and .x-only). Recount when a section changes a lot. */
const SECS={intro:['What is an LLM?',10,10,0],nn:['Neural networks in 15 minutes',15,15,0],why:['Why Transformers?',10,10,0],tokens:['Tokenization and embeddings',15,10,0],position:['Positional information',15,15,1],
  attention:['Self-attention from scratch',15,15,1],multihead:['Multi-head attention',10,10,1],block:['FFN, residuals and normalization',20,15,1],architecture:['Full decoder-only architecture',15,10,1],
  training:['Training an LLM',20,15,1],generation:['Generation and decoding',10,10,0],hardware:['GPUs and CUDA',20,15,1],inference:['Inference in practice',15,10,1],posttrain:['From base model to assistant',10,10,1],prefs:['Preference tuning and reasoning',15,10,1],
  encdec:['Encoder–decoder and the 2017 paper',25,15,1],modern:['What modern LLMs change',20,20,2]};
const LEVELS=['Foundation','Core','Deep dive'];
const TOPICS={'iv-core':'Transformer fundamentals','iv-rag':'RAG and embeddings','iv-ft':'Fine-tuning and alignment','iv-agents':'Agents and tool use','iv-prod':'Evaluation and production','iv-hw':'Hardware'};
const PATHS={full:{name:'Full depth',desc:'Exact worked examples, every interactive, the maths detail and the Python code.'},
  quick:{name:'Quick tour',desc:'Simplified worked examples; optional material is hidden. You can switch at any time.'}};
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
/* one saved record: path, done sections, correct quiz answers, last place, drill marks */
const store=(()=>{
  const K='tfs-progress';let d=null;
  try{d=JSON.parse(localStorage.getItem(K));}catch(e){}
  if(!d||typeof d!=='object')d={};
  d={path:d.path==='quick'?'quick':'full',done:d.done||{},quiz:d.quiz||{},last:d.last||null,drill:d.drill||{}};
  try{const old=localStorage.getItem('tfs-done');if(old){JSON.parse(old).forEach(id=>{if(SECS[id])d.done[id]=true;});localStorage.removeItem('tfs-done');}}catch(e){}
  return{d,save(){try{localStorage.setItem(K,JSON.stringify(d));}catch(e){}}};
})();
const P=store.d;
const minsOf=id=>SECS[id][P.path==='quick'?2:1];
const partStats=p=>{const n=p.secs.filter(id=>P.done[id]).length;return{n,total:p.secs.length,left:p.secs.filter(id=>!P.done[id]).reduce((a,id)=>a+minsOf(id),0)};};
const onPathChange=[];
function applyPath(){
  const quick=P.path==='quick';
  state.exact=!quick;document.body.classList.toggle('simple',quick);document.body.classList.toggle('ess',quick);
  $$('.pathBtn').forEach(b=>b.setAttribute('aria-pressed',b.dataset.p===P.path));
}
function setPath(p){P.path=p;store.save();applyPath();rerender();onPathChange.forEach(f=>f());}
function initSettings(){
  const menu=$('#setMenu');
  if(menu){const v=document.createElement('p');v.className='sver';v.textContent=SITE_VERSION.startsWith('__')?'Version: local copy':'Version '+SITE_VERSION;menu.append(v);}
  const root=document.documentElement;
  // Light or dark; until the reader chooses, the page follows the system setting
  const sysTheme=()=>window.matchMedia&&matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light';
  const markTheme=()=>{const t=root.getAttribute('data-theme')||sysTheme();$$('.thBtn').forEach(b=>b.setAttribute('aria-pressed',b.dataset.t===t));};
  const applyTheme=t=>{root.setAttribute('data-theme',t);try{localStorage.setItem('tfs-theme',t);}catch(e){}markTheme();};
  try{const t=localStorage.getItem('tfs-theme');if(t==='light'||t==='dark')root.setAttribute('data-theme',t);}catch(e){}
  markTheme();
  $$('.thBtn').forEach(b=>b.addEventListener('click',()=>applyTheme(b.dataset.t)));
  $$('.pathBtn').forEach(b=>b.addEventListener('click',()=>setPath(b.dataset.p)));
  const setBtn=$('#setBtn'),setMenu=$('#setMenu');
  const closeSet=ret=>{if(setMenu.hidden)return;setMenu.hidden=true;setBtn.setAttribute('aria-expanded','false');if(ret)setBtn.focus();};
  /* reset asks first in a modal <dialog>: it traps focus, closes on Escape and returns "reset" only from its button */
  const dlg=document.createElement('dialog');dlg.className='dlg';dlg.setAttribute('aria-labelledby','rsTitle');
  dlg.innerHTML='<form method="dialog"><h2 id="rsTitle">Reset progress?</h2><p>This clears the finished sections, quiz answers and drill marks saved in this browser. Your path, theme and code settings stay. You cannot undo this.</p>'
    +'<div class="acts"><button class="btn" value="cancel" autofocus>Cancel</button><button class="btn danger" value="reset">Reset progress</button></div></form>';
  document.body.append(dlg);
  dlg.addEventListener('click',e=>{if(e.target===dlg)dlg.close('cancel');});
  dlg.addEventListener('close',()=>{if(dlg.returnValue==='reset'){P.done={};P.quiz={};P.drill={};P.last=null;store.save();location.reload();}else setBtn.focus();});
  $('#pgReset').addEventListener('click',()=>{closeSet(false);dlg.returnValue='';dlg.showModal();});
  setBtn.addEventListener('click',()=>{if(setMenu.hidden){setMenu.hidden=false;setBtn.setAttribute('aria-expanded','true');const f=setMenu.querySelector('[aria-pressed="true"]')||setMenu.querySelector('button');if(f)f.focus();}else closeSet(true);});
  document.addEventListener('click',e=>{if(!setMenu.hidden&&!e.target.closest('.setwrap'))closeSet(false);});
  document.addEventListener('keydown',e=>{if(e.key==='Escape'&&!setMenu.hidden)closeSet(true);});
  setMenu.addEventListener('focusout',e=>{if(e.relatedTarget&&!e.relatedTarget.closest('.setwrap'))closeSet(false);});
  let rt=null,lastW=window.innerWidth;window.addEventListener('resize',()=>{if(window.innerWidth===lastW)return;lastW=window.innerWidth;clearTimeout(rt);rt=setTimeout(rerender,150);});
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
      $('.rt',s).innerHTML='Part '+L+' · '+(i+1)+' of '+secs.length+' · about '+minsOf(s.id)+' min <span class="lvl">'+LEVELS[SECS[s.id][3]]+'</span>';
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
  onPathChange.push(paint);

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
    const total=p=>Object.keys(SECS).reduce((a,id)=>a+SECS[id][p==='quick'?2:1],0);
    $('#paths').innerHTML=Object.keys(PATHS).map(k=>'<button type="button" class="pathCard" data-p="'+k+'" aria-pressed="'+(P.path===k)+'"><b>'+PATHS[k].name+'</b><span>'+PATHS[k].desc+'</span><small>About '+fmtMin(total(k))+' of study</small></button>').join('');
    $$('.pathCard').forEach(b=>b.addEventListener('click',()=>setPath(b.dataset.p)));
    const lastId=P.last&&P.last.split('#')[1];
    $('#resume').innerHTML=lastId&&SECS[lastId]?'<a class="btn pri big" href="'+P.last+'">Continue: '+SECS[lastId][0]+' →</a>':'<a class="btn pri big" href="part-a.html">Start with Part A →</a>';
    $('#journey').innerHTML=PARTS.map(p=>{const st=partStats(p),state=st.n===st.total?'complete':st.n?'started':'';
      return '<li class="'+state+'"><a class="jpart" href="'+partUrl(p)+'"><span class="jl">'+p.id.toUpperCase()+'</span><span class="jt"><b>'+p.name+'</b><span>'+p.goal+'</span>'
        +'<small>'+st.total+' sections · '+(state==='complete'?'complete':st.n?st.n+' done, about '+fmtMin(st.left)+' left':'about '+fmtMin(st.left))+'</small></span></a>'
        +'<ol class="jsecs">'+p.secs.map(id=>'<li class="'+(P.done[id]?'done':'')+'"><a href="'+partUrl(p)+'#'+id+'">'+SECS[id][0]+(P.done[id]?'<span class="vh">, done</span>':'')+'</a></li>').join('')+'</ol></li>';}).join('');
    const got=Object.values(P.drill).filter(v=>v==='got').length;
    if(got)$('#drillStat').textContent=got+' question'+(got>1?'s':'')+' marked “Got it” so far';
  }
  onPathChange.push(paint);paint();
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
  const chips=(k,opts)=>'<div class="chips" role="group" aria-label="'+(k==='topic'?'Topic':'Status')+'">'+opts.map(([v,t])=>'<button type="button" class="chip" data-k="'+k+'" data-v="'+v+'">'+t+'</button>').join('')+'</div>';
  $('#drillbar').innerHTML=chips('topic',[['all','All topics']].concat(Object.entries(TOPICS)))
    +chips('status',[['all','All'],['new','Not marked'],['again','Review again'],['got','Got it']])
    +'<p class="dcount" aria-live="polite"></p>';
  $('#drillbar').addEventListener('click',e=>{const b=e.target.closest('.chip');if(!b)return;f[b.dataset.k]=b.dataset.v;paint();});
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
    $$('#drillbar .chip').forEach(b=>{const on=f[b.dataset.k]===b.dataset.v;b.classList.toggle('on',on);b.setAttribute('aria-pressed',on);});
    const shown=qas.filter(q=>!q.hidden).length;
    $('.dcount').textContent=got+' of '+qas.length+' got it · '+again+' to review again'+(shown<qas.length?' · showing '+shown:'');
  }
  paint();
  const h=location.hash.slice(1);if(h&&document.getElementById(h))document.getElementById(h).scrollIntoView();
}

applyPath();
[initSettings,initCode,...PAGE_INITS,initXrefs,initPart,initHome,initDrill,initSelects].forEach(f=>{try{f();}catch(e){console.error(f.name,e);}});
renderTex(document);
