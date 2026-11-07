"use strict";
const http=require("node:http");
const fs=require("node:fs");
const path=require("node:path");
const {chromium}=require("playwright");
const manifest=JSON.parse(fs.readFileSync("private-browser-fixtures/manifest.json","utf8"));
const counters={};
const cases=[];
function norm(x){
 let v=(x||"").toLowerCase().replaceAll("_","-");
 if(["shift-jis","shift_jis","windows-31j","cp932"].includes(v))return "shift-jis";
 if(["gb2312","gbk","cp936"].includes(v))return "gbk";
 return v;
}
function bucket(group){
 if(!counters[group])counters[group]={n:0,browser_equals_html5lib:0,browser_equals_turbohtml:0,
   browser_equals_archive_label:0,browser_disagrees_both:0,
   where_parsers_disagree:0,where_parsers_agree:0};
 return counters[group];
}
async function main(){
 const server=http.createServer((req,res)=>{
   const id=Number((req.url||"").replace("/case/",""));
   if(!Number.isInteger(id)||id<0||id>=manifest.length){
     res.writeHead(404);res.end();return;
   }
   const entry=manifest[id];
   const filename=path.join("private-browser-fixtures",entry.file);
   res.writeHead(200,{"Content-Type":"text/html","Cache-Control":"no-store"});
   fs.createReadStream(filename).pipe(res);
 });
 await new Promise(resolve=>server.listen(0,"127.0.0.1",resolve));
 let browser;
 try{
   browser=await chromium.launch({headless:true,args:["--no-sandbox","--disable-dev-shm-usage"]});
   const context=await browser.newContext({javaScriptEnabled:false,serviceWorkers:"block"});
   const local="http://127.0.0.1:"+server.address().port;
   await context.route("**/*",route=>{
     if(route.request().url().startsWith(local+"/case/"))return route.continue();
     return route.abort();
   });
   const page=await context.newPage();
   for(const entry of manifest){
     await page.goto(local+"/case/"+entry.id,{waitUntil:"domcontentloaded",timeout:20000});
     const actual=await page.evaluate(()=>document.characterSet);
     const b=bucket(entry.folder);
     b.n++;
     b.browser_equals_html5lib+=+(norm(actual)==norm(entry.html5lib));
     b.browser_equals_turbohtml+=+(norm(actual)==norm(entry.turbohtml));
     b.browser_equals_archive_label+=+(norm(actual)==norm(entry.label));
     b.browser_disagrees_both+=+(norm(actual)!=norm(entry.html5lib) && norm(actual)!=norm(entry.turbohtml));
     b.where_parsers_disagree+=+entry.parsers_disagree;
     b.where_parsers_agree+=+(!entry.parsers_disagree);
     if(norm(entry.html5lib)!=norm(entry.turbohtml) && cases.length<60){
       cases.push({sha256:entry.sha256,folder:entry.folder,
        browser:actual,html5lib:entry.html5lib,turbohtml:entry.turbohtml});
     }
   }
   await page.close();
   await context.close();
   const result={
    schema:"P1-real-browser-encoding-arbitration-v1",
    source_count:manifest.length,
    design:"all HTML5lib-vs-turbohtml disagreement documents plus 12 parser-agreement controls per folder",
    actual_browser:"Playwright headless Chromium, JavaScript disabled, HTTP Content-Type text/html without charset, no external network requests",
    groups:counters,
    disagreement_examples_metadata_only:cases,
    notes:[
      "Selected disagreement-enriched sample, not random nor accuracy prevalence.",
      "Chromium differs by version and auto fallback locale, original archive lacks HTTP response headers.",
      "Browser document.characterSet reference only; source directory labels not verified truth.",
      "No source HTML, paths, text or decoded bytes exported."
    ],
    raw_html_exported:false,
   };
   fs.mkdirSync("research-results",{recursive:true});
   fs.writeFileSync("research-results/p1-real-browser-arbitration.json",JSON.stringify(result,null,2)+"\n");
   console.log("P1_BROWSER_ARBITRATION",JSON.stringify({source_count:manifest.length,groups:counters}));
 }finally{
   if(browser)await browser.close();
   await new Promise(resolve=>server.close(resolve));
 }
}
main().catch(e=>{console.error(e.stack||e);process.exitCode=1});
