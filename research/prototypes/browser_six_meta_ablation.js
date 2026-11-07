"use strict";
// Causal probe ONLY: ablate conflicting charset labels in six previously
// identified public historical HTML sources; never export the originals.
const fs=require("node:fs");
const http=require("node:http");
const {chromium}=require("playwright");
const rows=JSON.parse(fs.readFileSync("private-browser-fixtures/manifest.json","utf8"))
  .filter(e=>(e.folder==="GBK"||e.folder==="Windows-1256") && e.prototype_encoding==="utf-8");
if(rows.length!==6)throw new Error("unexpected selected disagreement count "+rows.length);
const variants=[];
for(const row of rows){
 const raw=fs.readFileSync("private-browser-fixtures/"+row.file).toString("latin1");
 for(const mode of ["original","ablate_utf8_label","ablate_other_label","ablate_both"]){
   let html=raw;
   let changeCount=0;
   const replace=(pattern)=>{
     html=html.replace(pattern,match=>{changeCount++;return "charset=invalid";});
   };
   if(mode==="ablate_utf8_label"||mode==="ablate_both")
     replace(/charset\s*=\s*["']?utf-8["']?/gi);
   if(mode==="ablate_other_label"||mode==="ablate_both")
     replace(/charset\s*=\s*["']?(?:gb2312|gbk|windows-1256)["']?/gi);
   variants.push({sha256:row.sha256,group:row.folder,mode,changeCount,bytes:Buffer.from(html,"latin1")});
 }
}
async function main(){
 const server=http.createServer((req,res)=>{
  const idx=Number((req.url||"").replace("/case/",""));
  if(!Number.isInteger(idx)||idx<0||idx>=variants.length){res.writeHead(404);res.end();return;}
  res.writeHead(200,{"Content-Type":"text/html","Cache-Control":"no-store"});
  res.end(variants[idx].bytes);
 });
 await new Promise(resolve=>server.listen(0,"127.0.0.1",resolve));
 let browser;
 try{
  browser=await chromium.launch({headless:true,args:["--no-sandbox","--disable-dev-shm-usage"]});
  const context=await browser.newContext({javaScriptEnabled:false,serviceWorkers:"block"});
  const base="http://127.0.0.1:"+server.address().port+"/case/";
  await context.route("**/*",route=>route.request().url().startsWith(base)?route.continue():route.abort());
  const page=await context.newPage();
  const reports=[];
  for(let i=0;i<variants.length;i++){
    const v=variants[i];
    await page.goto(base+i,{waitUntil:"domcontentloaded",timeout:20000});
    const browserEncoding=await page.evaluate(()=>document.characterSet);
    reports.push({sha256:v.sha256,group:v.group,mode:v.mode,
      replacements:v.changeCount,browser:browserEncoding});
  }
  console.log("P1_BROWSER_SIX_META_ABLATION",JSON.stringify(reports));
  await page.close();
  await context.close();
 }finally{
  if(browser)await browser.close();
  await new Promise(resolve=>server.close(resolve));
 }
}
main().catch(e=>{console.error(e);process.exitCode=1});
