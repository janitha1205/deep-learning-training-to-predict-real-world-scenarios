function [err,yh,yh2,out]=frwd(xin,xout,w1,w2,wo)
d_i=size(w1,1);
d_h=size(w2,1);
d_h2=size(wo,1);
d_o=size(wo,2);
ah=xin*w1;%[ah1 , ah2]
%activation function
yh=[];
for i=1:d_h
  yh=[yh,activation_f(ah(i))];
endfor
ah2=yh*w2;
yh2=[];
for i=1:d_h2
  yh2=[yh2,activation_f(ah2(i))];
endfor
a_out=yh2*wo;
out=[];
err=[];
for i=1:d_o
ou=activation_f(a_out(i));
%dou=ou*(1-ou)*(xout(i)-ou)
out=[out,ou];
err=[err,xout(i)-ou];
endfor
endfunction
