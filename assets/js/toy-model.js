'use strict';
/* A real (tiny) decoder-only Transformer with manual backprop, trained in the browser in Part C. No DOM access. */
/* ---------- Sections 8-9: a real (tiny) decoder-only Transformer with manual backprop ----------
   Simplifications versus a production GPT: character-level vocabulary, 2 layers, 1 attention head
   per layer, ReLU MLP, no LayerNorm, no dropout, no output-projection weight tying. */
const CORPUS={subj:['the cat','the dog','a bird','the fox'],verb:['sat on','ran to','saw','likes'],obj:['the mat','the log','a tree','the hat']};
const CHARS=Array.from(new Set((CORPUS.subj.concat(CORPUS.verb,CORPUS.obj).join('')+' .').split(''))).sort();
const STOI=new Map(CHARS.map((c,i)=>[c,i]));
function makeText(n,seed){const r=mulberry32(seed),pick=a=>a[Math.floor(r()*a.length)];let s='';for(let i=0;i<n;i++)s+=pick(CORPUS.subj)+' '+pick(CORPUS.verb)+' '+pick(CORPUS.obj)+'. ';return s;}
function encode(s){const out=[];for(const ch of s.toLowerCase()){if(STOI.has(ch))out.push(STOI.get(ch));}return out;}
function decode(a){return a.map(i=>CHARS[i]).join('');}
function matmul(X,W,n,inD,outD){const Y=new Float64Array(n*outD);for(let t=0;t<n;t++){const xo=t*inD,yo=t*outD;for(let i=0;i<inD;i++){const x=X[xo+i];if(x===0)continue;const wo=i*outD;for(let o=0;o<outD;o++)Y[yo+o]+=x*W[wo+o];}}return Y;}
function gradW(X,dY,n,inD,outD,G){for(let t=0;t<n;t++){const xo=t*inD,yo=t*outD;for(let i=0;i<inD;i++){const x=X[xo+i];if(x===0)continue;const wo=i*outD;for(let o=0;o<outD;o++)G[wo+o]+=x*dY[yo+o];}}}
function backX(dY,W,n,inD,outD,dX){for(let t=0;t<n;t++){const xo=t*inD,yo=t*outD;for(let i=0;i<inD;i++){const wo=i*outD;let s=0;for(let o=0;o<outD;o++)s+=dY[yo+o]*W[wo+o];dX[xo+i]+=s;}}return dX;}
function createModel(cfg,seed){
  const rng=mulberry32(seed),list=[];
  const P=(n,std)=>{const w=new Float64Array(n);if(std)for(let i=0;i<n;i++)w[i]=randn(rng)*std;const p={w:w,g:new Float64Array(n),m:new Float64Array(n),v:new Float64Array(n)};list.push(p);return p;};
  const V=cfg.V,T=cfg.T,d=cfg.d,m=cfg.m,sd=1/Math.sqrt(d);
  const p={E:P(V*d,0.3),P:P(T*d,0.3),layers:[]};
  for(let l=0;l<cfg.L;l++)p.layers.push({Wq:P(d*d,sd),Wk:P(d*d,sd),Wv:P(d*d,0.5*sd),W1:P(d*m,sd),b1:P(m,0),W2:P(m*d,0.5/Math.sqrt(m)),b2:P(d,0)});
  p.U=P(d*V,sd);p.bu=P(V,0);
  let count=0;for(const q of list)count+=q.w.length;
  return {cfg:cfg,p:p,list:list,count:count,t:0};
}
function forward(model,toks){
  const cfg=model.cfg,V=cfg.V,d=cfg.d,m=cfg.m,n=toks.length,p=model.p,sc=1/Math.sqrt(d);
  let X=new Float64Array(n*d);
  for(let t=0;t<n;t++)for(let k=0;k<d;k++)X[t*d+k]=p.E.w[toks[t]*d+k]+p.P.w[t*d+k];
  const layers=[];
  for(const lp of p.layers){
    const Q=matmul(X,lp.Wq.w,n,d,d),K=matmul(X,lp.Wk.w,n,d,d),Vv=matmul(X,lp.Wv.w,n,d,d);
    const A=new Float64Array(n*n),H=new Float64Array(X);
    for(let i=0;i<n;i++){
      let mx=-Infinity;
      for(let j=0;j<=i;j++){let s=0;for(let k=0;k<d;k++)s+=Q[i*d+k]*K[j*d+k];s*=sc;A[i*n+j]=s;if(s>mx)mx=s;}
      let sum=0;for(let j=0;j<=i;j++){const e=Math.exp(A[i*n+j]-mx);A[i*n+j]=e;sum+=e;}
      for(let j=0;j<=i;j++){const a=A[i*n+j]/sum;A[i*n+j]=a;for(let k=0;k<d;k++)H[i*d+k]+=a*Vv[j*d+k];}
    }
    const Z=matmul(H,lp.W1.w,n,d,m);for(let t=0;t<n;t++)for(let k=0;k<m;k++)Z[t*m+k]+=lp.b1.w[k];
    const R=Z.map(v=>v>0?v:0);
    const X2=matmul(R,lp.W2.w,n,m,d);for(let t=0;t<n;t++)for(let k=0;k<d;k++)X2[t*d+k]+=lp.b2.w[k]+H[t*d+k];
    layers.push({X:X,Q:Q,K:K,Vv:Vv,A:A,H:H,Z:Z,R:R});X=X2;
  }
  const logits=matmul(X,p.U.w,n,d,V);for(let t=0;t<n;t++)for(let v=0;v<V;v++)logits[t*V+v]+=p.bu.w[v];
  return {n:n,toks:toks,layers:layers,Xf:X,logits:logits};
}
function seqLoss(model,c,targets){const V=model.cfg.V;let loss=0;for(let t=0;t<c.n;t++){const pr=softmax(Array.from(c.logits.subarray(t*V,t*V+V)));loss-=Math.log(pr[targets[t]]+1e-12);}return loss;}
function backward(model,c,targets,scale){
  const cfg=model.cfg,V=cfg.V,d=cfg.d,m=cfg.m,n=c.n,p=model.p,sc=1/Math.sqrt(d);
  const dl=new Float64Array(n*V);let loss=0;
  for(let t=0;t<n;t++){const pr=softmax(Array.from(c.logits.subarray(t*V,t*V+V)));loss-=Math.log(pr[targets[t]]+1e-12);for(let v=0;v<V;v++){const g=(pr[v]-(v===targets[t]?1:0))*scale;dl[t*V+v]=g;p.bu.g[v]+=g;}}
  gradW(c.Xf,dl,n,d,V,p.U.g);
  let dX=backX(dl,p.U.w,n,d,V,new Float64Array(n*d));
  for(let l=p.layers.length-1;l>=0;l--){
    const lc=c.layers[l],lp=p.layers[l];
    gradW(lc.R,dX,n,m,d,lp.W2.g);for(let t=0;t<n;t++)for(let k=0;k<d;k++)lp.b2.g[k]+=dX[t*d+k];
    const dZ=backX(dX,lp.W2.w,n,m,d,new Float64Array(n*m));
    for(let i=0;i<dZ.length;i++)if(lc.Z[i]<=0)dZ[i]=0;
    gradW(lc.H,dZ,n,d,m,lp.W1.g);for(let t=0;t<n;t++)for(let k=0;k<m;k++)lp.b1.g[k]+=dZ[t*m+k];
    const dH=backX(dZ,lp.W1.w,n,d,m,new Float64Array(dX));
    const dQ=new Float64Array(n*d),dK=new Float64Array(n*d),dV=new Float64Array(n*d),da=new Float64Array(n);
    for(let i=0;i<n;i++){
      let mean=0;
      for(let j=0;j<=i;j++){let s=0;for(let k=0;k<d;k++)s+=dH[i*d+k]*lc.Vv[j*d+k];da[j]=s;mean+=lc.A[i*n+j]*s;}
      for(let j=0;j<=i;j++){const a=lc.A[i*n+j],ds=a*(da[j]-mean)*sc;for(let k=0;k<d;k++){dV[j*d+k]+=a*dH[i*d+k];dQ[i*d+k]+=ds*lc.K[j*d+k];dK[j*d+k]+=ds*lc.Q[i*d+k];}}
    }
    gradW(lc.X,dQ,n,d,d,lp.Wq.g);gradW(lc.X,dK,n,d,d,lp.Wk.g);gradW(lc.X,dV,n,d,d,lp.Wv.g);
    backX(dQ,lp.Wq.w,n,d,d,dH);backX(dK,lp.Wk.w,n,d,d,dH);backX(dV,lp.Wv.w,n,d,d,dH);
    dX=dH;
  }
  for(let t=0;t<n;t++)for(let k=0;k<d;k++){p.E.g[c.toks[t]*d+k]+=dX[t*d+k];p.P.g[t*d+k]+=dX[t*d+k];}
  return loss;
}
function adamStep(model,lr){
  const b1=0.9,b2=0.99,eps=1e-8;model.t++;
  let norm=0;for(const q of model.list)for(let i=0;i<q.g.length;i++)norm+=q.g[i]*q.g[i];
  norm=Math.sqrt(norm);const clip=norm>5?5/norm:1,c1=1-Math.pow(b1,model.t),c2=1-Math.pow(b2,model.t);
  for(const q of model.list)for(let i=0;i<q.w.length;i++){const g=q.g[i]*clip;q.m[i]=b1*q.m[i]+(1-b1)*g;q.v[i]=b2*q.v[i]+(1-b2)*g*g;q.w[i]-=lr*(q.m[i]/c1)/(Math.sqrt(q.v[i]/c2)+eps);q.g[i]=0;}
  return norm;
}
const TOY_CFG={T:16,d:24,m:48,L:2,B:8};
function createTrainer(o){
  const train=encode(makeText(o.nTrain,o.seed+1)),val=encode(makeText(40,o.seed+2));
  return {model:createModel({V:CHARS.length,T:TOY_CFG.T,d:TOY_CFG.d,m:TOY_CFG.m,L:TOY_CFG.L},o.seed),train:train,val:val,trainText:decode(train),step:0,rng:mulberry32(o.seed+3)};
}
function trainStep(tr,lr){
  const T=tr.model.cfg.T,B=TOY_CFG.B;let loss=0;
  for(let b=0;b<B;b++){const s=Math.floor(tr.rng()*(tr.train.length-T-1));const c=forward(tr.model,tr.train.slice(s,s+T));loss+=backward(tr.model,c,tr.train.slice(s+1,s+T+1),1/(B*T));}
  adamStep(tr.model,lr);tr.step++;return loss/(B*T);
}
function evalLoss(model,data){
  const T=model.cfg.T,nWin=20,span=data.length-T-1;let loss=0;
  for(let w=0;w<nWin;w++){const s=Math.floor(w*span/nWin);loss+=seqLoss(model,forward(model,data.slice(s,s+T)),data.slice(s+1,s+T+1));}
  return loss/(nWin*T);
}
function lrAt(step,start,total,base){const span=Math.max(1,total-start),k=step-start,warm=Math.max(1,Math.floor(0.05*span));if(k<warm)return base*(k+1)/warm;const f=Math.min(1,(k-warm)/Math.max(1,span-warm));return base*(0.1+0.9*0.5*(1+Math.cos(Math.PI*f)));}
function nextLogits(model,ctx){const T=model.cfg.T,V=model.cfg.V,win=ctx.slice(-T),c=forward(model,win);return Array.from(c.logits.subarray((win.length-1)*V,win.length*V));}
function decodeDist(logits,o){
  const V=logits.length,pT=softmax(logits,Math.max(o.temp,0.05));
  const order=pT.map((_,i)=>i).sort((a,b)=>pT[b]-pT[a]),cut=new Array(V).fill('');
  order.forEach((i,r)=>{if(r>=o.topK)cut[i]='k';});
  if(o.topP<1){let cum=0;for(const i of order){if(cut[i])continue;if(cum>=o.topP)cut[i]='p';cum+=pT[i];}}
  let s=0;const pF=pT.map((q,i)=>cut[i]?0:q);for(const q of pF)s+=q;
  return {pT:pT,pF:pF.map(q=>q/s),cut:cut,order:order};
}
function generate(model,prompt,n,temp,seed){const rng=mulberry32(seed),ctx=encode(prompt);if(!ctx.length)ctx.push(STOI.get(' '));for(let i=0;i<n;i++)ctx.push(sampleFrom(softmax(nextLogits(model,ctx),Math.max(temp,0.05)),rng));return decode(ctx);}
