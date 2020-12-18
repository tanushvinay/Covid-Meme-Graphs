#Creating Dataset

We created a dataset for training the Graphs classifier.

The images were obtained using Google search and by using the following JavaScript snippet in the Browser's Developer Console

``` var cont=document.getElementsByTagName("body")[0]; var imgs=document.getElementsByTagName("a"); var i=0;var divv= document.createElement("div"); var aray=new Array();var j=-1; while(++i<imgs.length){ if(imgs[i].href.indexOf("/imgres?imgurl=http")>0){ divv.appendChild(document.createElement("br")); aray[++j]=decodeURIComponent(imgs[i].href).split(/=|%|&/)[1].split("?imgref")[0]; divv.appendChild(document.createTextNode(aray[j])); } } cont.insertBefore(divv,cont.childNodes[0]); ```
