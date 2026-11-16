'use strict';
/* Reference page: the glossary filter. */
/* ================= section and widget helpers ================= */
function initGloss(){$('#gl-q').addEventListener('input',e=>{const q=e.target.value.trim().toLowerCase();let n=0;$$('#gloss .term').forEach(t=>{const hit=!q||t.textContent.toLowerCase().indexOf(q)>=0;t.hidden=!hit;if(hit)n++;});$('#gl-none').hidden=n>0;});}

PAGE_INITS.push(initGloss);
