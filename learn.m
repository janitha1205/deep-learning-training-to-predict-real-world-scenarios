function [w1,w2,wo]=learn(out,xout,yh,wo,xin,w1,yh2,w2)
%xin=rand(1,d_i);
%xout=rand(d_o);
%w1=rand(d_i,d_h2);
%w2=rand(d_h2,d_h);
%wo=rand(d_h,d_o);
d_i=size(w1,1);
d_h=size(w1,1);
d_h2=size(w2,1);
d_o=size(wo,2);
%back propergation
dou=[];
lr=0.91;
dwo=[];
%output layer
for i=1:d_o
ou=out(i);
dou1=ou*(1-ou)*(xout(i)-ou);
dou=[dou,dou1];
endfor
%out to hidden layer
nu=1;
whh=[];
for k=1:d_h
  wh=0;
  for i=1:d_o
    wo(k,i)+=nu*yh(k)*dou(i);
    wh+=wo(k,i)*dou(i);
  endfor
  whh=[whh,yh(k)*(1-yh(k))*wh];
endfor
whh2=[];
%hidden layer 2
for k=1:d_h2
  wh2=0;
  for j=1:d_h
     dw=nu*yh2(k)*whh(j);
     w2(k,j)+=dw;
     wh2+=w2(k,j)*yh2(k);
  endfor
  whh2=[whh2,yh2(k)*(1-yh2(k))*wh2];
endfor
%hidden layer 1

for k=1:d_i
  for j=1:d_h2
     dw=nu*xin(k)*whh2(j);
     w1(k,j)+=dw;
  endfor
endfor
endfunction
