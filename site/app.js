function bars(id,rows,fmt,color){const el=document.getElementById(id),mx=Math.max(...rows.map(r=>r[1]));el.innerHTML=rows.map(([k,v])=>`<div class="bar"><span>${k}</span><i style="width:${Math.max(2,v/mx*100)}%;background:${color||''}"></i><span>${fmt?fmt(v):v.toLocaleString()}</span></div>`).join('')}
const pct=v=>(v*100).toFixed(1)+'%';
bars('fp',[['Before threshold',0.3792],['After conf ≥ 0.5',0.0138]],pct,'#ff5a5f');
bars('dup',[['Dup box per frame',0.0009],['Dup track candidates',0.1339],['Embedding coverage',0.1904]],pct,'#ffd166');
bars('bake',[['gemma4:26b',0.95],['qwen3.6:35b',1.15],['qwen3.8:27b',3.30],['gemma3:27b',4.11]],v=>v.toFixed(2)+' s','#c77dff');
fetch('assets/data.json').then(r=>r.json()).then(d=>{
 bars('labels',d.labels);bars('kinds',d.kinds,null,'#80ed99');
 document.getElementById('gallery').innerHTML=d.assets.map(a=>`<figure><img loading="lazy" src="assets/${a.file}" alt="${a.caption}"><figcaption>${a.caption}<br>${a.dets.map(t=>`<span class="tag">${t}</span>`).join('')}</figcaption></figure>`).join('');
 const s=d.stress,rows=[['Candidates screened',s.candidates],['VLM',s.vlm_model],['VLM screening total',s.vlm_total_s+' s'],['VLM latency p50 / p95',s.vlm_p50_s+' s / '+s.vlm_p95_s+' s'],['Throughput',s.imgs_per_min+' images/min · '+s.tok_per_s+' tok/s'],['Model resident memory',s.ollama_resident_gb+' GB (GPU)'],['Passed privacy screen',s.passed_privacy+' of '+s.candidates],['Selected after human review',s.selected],['Annotate frames',s.annotate_s+' s'],['Encode MP4 / WebM',s.encode_mp4_s+' s / '+s.encode_webm_s+' s'],['Production slow loop (historic)',s.hist_gemma4_ms+' ms VLM avg over '+s.hist_gemma4_n.toLocaleString()+' keyframes']];
 document.getElementById('stress').innerHTML=rows.map(r=>`<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join('')});
