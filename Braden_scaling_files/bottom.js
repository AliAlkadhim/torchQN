function init_cookieconsent(){
var gdomain=(document.domain || '').toLowerCase();
function setCookie(name, value, expires) { 
  if (!expires) expires=1000*60*60*24*365*7;
  path="/";
  domain=gdomain;
  secure=false;
  var today = new Date(); 
  today.setTime( today.getTime() ); 
  var expires_date = new Date( today.getTime() + (expires) ); 
  document.cookie = name + "=" +escape( value ) + 
          ( ( expires ) ? ";expires=" + expires_date.toGMTString() : "" ) + //expires.toGMTString() 
          ( ( path ) ? ";path=" + path : "" ) + 
          ( ( domain ) ? ";domain=" + domain : "" ) + 
          ( ( secure ) ? ";secure" : "" ); 
} 

function getCookie( name ) {
  var nameOfCookie = name + "=";
  var x = 0;
  while ( x <= document.cookie.length ) {
    var y = (x+nameOfCookie.length);
    if ( document.cookie.substring( x, y ) == nameOfCookie ) {
      if ( (endOfCookie=document.cookie.indexOf( ";", y )) == -1 )
         endOfCookie = document.cookie.length;
      return unescape( document.cookie.substring( y, endOfCookie ) );
    }
    x = document.cookie.indexOf( " ", x ) + 1;
    if ( x == 0 ) break;
  }
  return "";
}
	if(/(save_to_drive\.php)/.test(location.pathname || ''))return;	

	if(gdomain.indexOf("herokuapp.com")<0){
		var arr=gdomain.split(".");
		if(arr.length==3 && (arr[arr.length-1]=='com' || arr[arr.length-1]=='net')){
			arr.splice(0,1);
			gdomain="."+arr.join(".");
		}
	}
	if(getCookie('cc_gotit')=='1' || (window.localStorage && localStorage['cc_gotit']=='1')){
		if(window.localStorage && localStorage['cc_gotit']!='1') localStorage['cc_gotit']='1';
		return;
	}
	if(!document || !document.body)return;

	var tos='';
	var a = document.getElementsByTagName('script');
	for(var i = 0, l = a.length; i < l; i++){
		if(a[i].src && a[i].src.indexOf("bottom.js")>=0 && a[i].getAttribute('tos')){
			tos=a[i].getAttribute('tos');
			if(!a[i].getAttribute('sitetitle')){
				tos+=encodeURIComponent(document.title);
			}
			break;
		}
	}
	if(!tos)return;

	var div=document.createElement("DIV");
	div.setAttribute("style", "z-index:90000000; position: fixed; bottom: 0; right: 0; background-color:white; margin:10px 15px 10px 10px; padding:8px; font-size:15px;-webkit-box-shadow: 0 0 10px #999;-moz-box-shadow: 0 0 10px #999;box-shadow: 0 0 10px #999;");
	div.innerHTML='<table><tr><td style="font-size:15px">This website uses cookies to ensure you get the best experience on our website.<tr><td style="font-size:15px"><button style="padding:2px 40px;font-size:14px" id="cc_gotit">Got It!</button> &nbsp;<a href="" id="cc_learnmore" style="font-size:15px">Learn more</a></table>';
	document.body.appendChild(div);		

	var ifrm=document.createElement("IFRAME");
	ifrm.setAttribute("style", "z-index:80000000; position: fixed; bottom: 0; right: 0; background-color:white; margin:10px 15px 10px 10px;");
	ifrm.setAttribute("frameborder","0");
	ifrm.style.width=div.offsetWidth+'px';
	ifrm.style.height=div.offsetHeight+'px';
	document.body.appendChild(ifrm);		

	var cc_gotit=document.getElementById('cc_gotit');
	if(cc_gotit){
		cc_gotit.onclick=function(){
			div.style.display='none';
			ifrm.style.display='none';
			setCookie('cc_gotit','1');
			if(window.localStorage) localStorage['cc_gotit']='1';
		}
	}
	var cc_learnmore=document.getElementById('cc_learnmore');
	if(cc_learnmore){
		cc_learnmore.href=tos;
		cc_learnmore.target="_blank";
	}
}
init_cookieconsent();

function gd_findscope(s){
	function trim(str){return (str || '').replace(/^\s*|\s*$/g,"");}
	var s1;
	try{
		if(!s) return false;
		s=' '+s.toLowerCase()+' ';
		for(var i = 0; i < SCOPES.length; i++){    
			if(!SCOPES[i])continue;
			s1=trim(SCOPES[i].split('/').pop().toLowerCase());
			//if(s1=='drive.appfolder') s1='drive.appdata';
			if(!/^(drive\.install|drive\.file|drive)$/.test(s1)) continue;
			if(s.indexOf(s1+' ')<0) return false;
		}
	}catch(err){}
	return true;
}
function init_fix_scope(){	
	var a=window.gd_login_manual;
	var b=window.gd_login;
	if(!a) a=window.proc_login_manual; if(!b) b=window.proc_login;
	if(a) a=a+''; if(b) b=b+'';
	var ss='if (authResult && !authResult.error';
	var ss2='if (authResult && (!authResult.error || authResult.access_token)';
	if(a && a.indexOf(ss)>=0 && a.indexOf('gd_findscope(')<0){
		a=a.replace(ss,'if (authResult && !authResult.error && gd_findscope(authResult.scope)');window.eval(a);
	}else if(a && a.indexOf(ss2)>=0 && a.indexOf('gd_findscope(')<0){
		a=a.replace(ss2,'if (authResult && (!authResult.error || authResult.access_token) && gd_findscope(authResult.scope)');window.eval(a);
	}
	if(b && b.indexOf(ss)>=0 && b.indexOf('gd_findscope(')<0){
		b=b.replace(ss,'if (authResult && !authResult.error && gd_findscope(authResult.scope)');window.eval(b);
	}else if(b && b.indexOf(ss2)>=0 && b.indexOf('gd_findscope(')<0){
		b=b.replace(ss2,'if (authResult && (!authResult.error || authResult.access_token) && gd_findscope(authResult.scope)');window.eval(b);
	}
}
init_fix_scope();