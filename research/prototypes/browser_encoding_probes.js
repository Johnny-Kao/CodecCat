"use strict";
// Browser-only counterfactual probes to test fallback rules on unlabeled bytes.
// No fetched pages, JavaScript disabled in page context, server loopback only.
const http=require("node:http");
const {chromium}=require("playwright");
const examples=[
  {name:"ascii-only-no-meta",html:Buffer.from("<p>Hello world.</p>")},
  {name:"utf8-no-meta-chinese",html:Buffer.from("<p>中文資訊與研究資料：字元編碼偵測是問題。</p>","utf8")},
  {name:"utf8-no-meta-japanese",html:Buffer.from("<p>日本語の文字エンコーディングを確認する。</p>","utf8")},
  {name:"invalid-utf8-cp1252",html:Buffer.from("<p>caf\xe9 cr\xe8me</p>","latin1")},
  {name:"fake-utf8-script-then-gbk",
    html:Buffer.concat([Buffer.from('<script>var s="<meta charset=utf-8>"</script><meta charset=gbk><p>'),Buffer.from([0xd6,0xd0,0xce,0xc4]),Buffer.from("</p>")])},
  {name:"valid-utf8-head-then-gbk-head",
    html:Buffer.concat([Buffer.from('<meta charset=utf-8><meta charset=gbk><p>'),Buffer.from([0xd6,0xd0,0xce,0xc4]),Buffer.from("</p>")])},
  {name:"utf8-body-then-gbk-body",
    html:Buffer.concat([Buffer.from('<html><body><meta charset=utf-8><meta charset=gbk><p>'),Buffer.from([0xd6,0xd0,0xce,0xc4]),Buffer.from("</p>")])},
];
async function main(){
 const server=http.createServer((req,res)=>{
   const index=Number((req.url||"").replace("/probe/",""));
   if(!Number.isInteger(index)||index<0||index>=examples.length){res.writeHead(404);res.end();return;}
   res.writeHead(200,{"Content-Type":"text/html","Cache-Control":"no-store"});
   res.end(examples[index].html);
 });
 await new Promise(resolve=>server.listen(0,"127.0.0.1",resolve));
 let browser;
 try{
   browser=await chromium.launch({headless:true,args:["--no-sandbox","--disable-dev-shm-usage"]});
   const context=await browser.newContext({javaScriptEnabled:false,serviceWorkers:"block"});
   const base="http://127.0.0.1:"+server.address().port+"/probe/";
   await context.route("**/*",route=>route.request().url().startsWith(base)?route.continue():route.abort());
   const result=[];
   for(let i=0;i<examples.length;i++){
     const p=await context.newPage();
     await p.goto(base+i,{waitUntil:"domcontentloaded"});
     const selected=await p.evaluate(()=>document.characterSet);
     result.push({case:examples[i].name,browser:selected});
     await p.close();
   }
   console.log("P1_BROWSER_TARGETED_FALLBACK_PROBES",JSON.stringify(result));
   await context.close();
 }finally{
   if(browser)await browser.close();
   await new Promise(resolve=>server.close(resolve));
 }
}
main().catch(e=>{console.error(e);process.exitCode=1});
