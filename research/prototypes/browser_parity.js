"use strict";
// Browser arbitration of opt-in research HTML decoding, isolated from network.
// The private original HTML files and manifest must never become artifacts.
const fs=require("node:fs");
const http=require("node:http");
const path=require("node:path");
const crypto=require("node:crypto");
const {chromium}=require("playwright");

const entries=JSON.parse(fs.readFileSync("private-browser-fixtures/manifest.json","utf8"));
function normalized(label){
 const s=String(label||"").toLowerCase().replaceAll("_","-");
 if(["cp932","windows-31j","shift-jis"].includes(s))return "shift_jis";
 if(["gb2312","cp936","gbk"].includes(s))return "gbk";
 if(["cp1251","windows-1251"].includes(s))return "windows-1251";
 if(["cp1256","windows-1256"].includes(s))return "windows-1256";
 if(["cp1252","windows-1252","iso-8859-1"].includes(s))return "windows-1252";
 return s;
}
function digest(s){
 return crypto.createHash("sha256").update(s,"utf8").digest("hex");
}
function countsFor(obj,key){
 if(!obj[key])obj[key]={
  n:0,prototype_browser_encoding_match:0,prototype_browser_encoding_miss:0,
  prototype_browser_unicode_match:0,prototype_browser_unicode_miss:0,
  html5lib_browser_encoding_match:0,turbohtml_browser_encoding_match:0,
  prototype_fallback_cases:0,prototype_meta_cases:0,
  disagreement_selected_cases:0,agreement_control_cases:0,
  cp932_unicode_browser_matches:0,cp932_unicode_browser_misses:0,
 };
 return obj[key];
}
async function main(){
 const server=http.createServer((req,res)=>{
   const id=Number((req.url||"").replace("/case/",""));
   if(!Number.isInteger(id)||id<0||id>=entries.length){res.writeHead(404);res.end();return;}
   res.writeHead(200,{"Content-Type":"text/html","Cache-Control":"no-store"});
   fs.createReadStream(path.join("private-browser-fixtures",entries[id].file)).pipe(res);
 });
 await new Promise(ok=>server.listen(0,"127.0.0.1",ok));
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
  const groups={}, examples=[];
  for(const e of entries){
    await page.goto(local+"/case/"+e.id,{waitUntil:"domcontentloaded",timeout:20000});
    const browserEncoding=await page.evaluate(()=>document.characterSet);
    const data=fs.readFileSync(path.join("private-browser-fixtures",e.file));
    const expectedText=new TextDecoder(browserEncoding,{fatal:false}).decode(data);
    const sameEncoding=normalized(browserEncoding)===normalized(e.prototype_encoding);
    const sameUnicode=digest(expectedText)===e.prototype_text_sha256;
    const cp932UnicodeMatch=e.cp932_unicode_sha256!==null
      && digest(expectedText)===e.cp932_unicode_sha256;
    const domMetadata=await page.evaluate(()=>({
      meta_charset_labels:Array.from(document.querySelectorAll("meta[charset]"))
        .slice(0,8).map(m=>m.getAttribute("charset")),
      meta_equiv_labels:Array.from(document.querySelectorAll("meta[http-equiv]"))
        .slice(0,8).map(m=>m.getAttribute("content")),
      meta_locations:Array.from(document.querySelectorAll("meta"))
        .filter(m=>m.hasAttribute("charset")||/content-type/i.test(m.getAttribute("http-equiv")||""))
        .slice(0,8).map(m=>({
          parent_tag:m.parentElement?.tagName||null,
          in_head:!!m.closest("head"),
          in_body:!!m.closest("body"),
          content_type:(m.getAttribute("content")||"").slice(0,90),
          charset:(m.getAttribute("charset")||"").slice(0,60),
        })),
      head_meta_count:document.head?.querySelectorAll("meta").length||0,
      body_meta_count:document.body?.querySelectorAll("meta").length||0,
    }));
    const group=countsFor(groups,e.folder);
    group.n++;
    group.prototype_browser_encoding_match+=Number(sameEncoding);
    group.prototype_browser_encoding_miss+=Number(!sameEncoding);
    group.prototype_browser_unicode_match+=Number(sameUnicode);
    group.prototype_browser_unicode_miss+=Number(!sameUnicode);
    if(e.cp932_unicode_sha256!==null){
      group.cp932_unicode_browser_matches+=Number(cp932UnicodeMatch);
      group.cp932_unicode_browser_misses+=Number(!cp932UnicodeMatch);
    }
    group.html5lib_browser_encoding_match+=Number(normalized(browserEncoding)===normalized(e.html5lib));
    group.turbohtml_browser_encoding_match+=Number(normalized(browserEncoding)===normalized(e.turbohtml));
    group.prototype_fallback_cases+=Number(e.prototype_source==="fallback");
    group.prototype_meta_cases+=Number(e.prototype_source==="meta");
    group.disagreement_selected_cases+=Number(e.parsers_disagree);
    group.agreement_control_cases+=Number(!e.parsers_disagree);
    if((!sameEncoding || !sameUnicode) && examples.length<40){
      examples.push({sha256:e.sha256,group:e.folder,
        browser_encoding:browserEncoding,prototype_encoding:e.prototype_encoding,
        source:e.prototype_source,encoding_match:sameEncoding,unicode_match:sameUnicode,
        cp932_unicode_match:e.cp932_unicode_sha256===null?null:cp932UnicodeMatch,
        utf8_strict_valid:e.utf8_strict_valid,
        first_charset_ascii_offset:e.first_charset_ascii_offset,
        meta_marker_count:e.meta_marker_count,
        html5lib_selected:e.html5lib,turbohtml_selected:e.turbohtml,
        parsed_meta_labels:domMetadata.meta_charset_labels,
        parsed_http_equiv:domMetadata.meta_equiv_labels,
        meta_locations:domMetadata.meta_locations,
        head_meta_count:domMetadata.head_meta_count,
        body_meta_count:domMetadata.body_meta_count});
    }
  }
  const aggregate={
   schema:"codeccat-html-aware-prototype-chromium-v1",
   selected_cases:entries.length,
   selection:"all 190 previous parser disagreements and 60 agreement controls, source screened",
   groups,
   mismatches_metadata_only:examples,
   caveats:[
    "Browser encoding label + WHATWG TextDecoder output as operational reference, not verified authored Unicode truth.",
    "HTTP Content-Type text/html without charset; unknown original headers or source locale.",
    "This is a disagreement-enriched sample; no claim of general population correctness.",
    "Codecs in webencodings use CPython mappings; WHATWG decoding mappings can differ for Shift_JIS.",
    "Prototype is not part of installed CodecCat; no model or default detection changes."
   ],
   external_resources_blocked:true,
   javascript_disabled:true,
   raw_html_exported:false
  };
  fs.mkdirSync("research-results",{recursive:true});
  fs.writeFileSync("research-results/html-aware-prototype-browser-parity.json",JSON.stringify(aggregate,null,2)+"\n");
  console.log("P1_HTML_PROTOTYPE_BROWSER_PARITY",JSON.stringify({n:entries.length,groups}));
  console.log("P1_HTML_PROTOTYPE_MISMATCH_DIAGNOSIS",JSON.stringify(examples));
  await page.close();
  await context.close();
 }finally{
  if(browser)await browser.close();
  await new Promise(resolve=>server.close(resolve));
 }
}
main().catch(e=>{console.error(e.stack||e);process.exitCode=1});
