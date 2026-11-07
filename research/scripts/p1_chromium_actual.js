"use strict";
// Research-only true Chromium HTML encoding results on exact original fixture bytes.
const http=require("node:http");
const {chromium}=require("playwright");
const cases=[
 ["plain-meta",Buffer.concat([Buffer.from('<meta charset=gbk><p>'),Buffer.from([0xd6,0xd0,0xce,0xc4]),Buffer.from("</p>")]),null],
 ["http-equiv",Buffer.concat([Buffer.from('<meta http-equiv="Content-Type" content="text/html; charset=gbk"><p>'),Buffer.from([0xd6,0xd0]),Buffer.from("</p>")]),null],
 ["comment-meta",Buffer.from('<!-- <meta charset=gbk> --><p>hello</p>'),null],
 ["script-fake-meta",Buffer.from('<script>var html="<meta charset=gbk>";</script><p>hello</p>'),null],
 ["attribute-fake-meta",Buffer.from('<div data-value="<meta charset=gbk>">Hi</div>'),null],
 ["meta-after-1024",Buffer.concat([Buffer.from("X".repeat(1024)),Buffer.from('<meta charset=gbk><p>'),Buffer.from([0xd6,0xd0]),Buffer.from("</p>")]),null],
 ["transport-wins",Buffer.from('<meta charset=gbk><p>hello</p>'),"utf-8"],
 ["bom-wins",Buffer.concat([Buffer.from([0xef,0xbb,0xbf]),Buffer.from('<meta charset=gbk><p>hello</p>')]),null],
 ["doubled-meta",Buffer.from('<meta charset=gbk><meta charset=utf-8><p>text</p>'),null],
 ["quoted-angle",Buffer.from('<meta title="> embedded" charset=gbk><p>text</p>'),null],
];
async function main(){
 const server=http.createServer((req,res)=>{
  const index=Number((req.url||"").replace("/case/",""));
  if(!Number.isInteger(index) || index<0||index>=cases.length){res.writeHead(404);res.end();return;}
  const [name,raw,charset]=cases[index];
  res.writeHead(200,{"Content-Type":"text/html"+(charset?"; charset="+charset:""),"Cache-Control":"no-store"});
  res.end(raw);
 });
 await new Promise(resolve=>server.listen(0,"127.0.0.1",resolve));
 let browser;
 try {
  browser=await chromium.launch({headless:true,args:["--no-sandbox","--disable-dev-shm-usage"]});
  const results=[];
  for (let i=0;i<cases.length;i++){
    const page=await browser.newPage();
    await page.goto("http://127.0.0.1:"+server.address().port+"/case/"+i,{waitUntil:"domcontentloaded"});
    const result=await page.evaluate(()=>({
     charset:document.characterSet,
     declared:document.querySelector("meta[charset]")?.getAttribute("charset")||null,
     bodyText:document.body?.textContent?.slice(-24)||""
    }));
    results.push({name:cases[i][0],charset:result.charset,declaredMeta:result.declared,
                  note:"browser parsing+sniff result"});
    await page.close();
  }
  console.log("P1_CHROMIUM_ACTUAL",JSON.stringify(results));
 }finally{
  if(browser)await browser.close();
  await new Promise(resolve=>server.close(resolve));
 }
}
main().catch(e=>{console.error(e);process.exitCode=1});
