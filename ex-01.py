d_i=2;
d_o=1;
d_h=20;
d_h2=40;
n=10;
xin=rand(n,d_i);

xout=rand(n,d_o);

w1=rand(d_i,d_h);
w2=rand(d_h,d_h2);
wo=rand(d_h2,d_o);
% [[w11,w21],[w12,w22]]
%forward pass
epoch=100;
err=[];
id=[];
err=[];
out=[];
yh1=[];
yh2=[];
errm=[];
for i=1:n
[ou,yh11,yh21,out1]=frwd(xin(i,:),xout(i,:),w1,w2,wo);
err=[err;ou];
yh1=[yh1;yh11];
yh2=[yh2;yh21];
out=[out;out1];
endfor
for i=1:epoch
  w11=0;
  w21=0;
  wo1=0;
  for k=1:n
    [w3,w4,wo5]=learn(out(k),xout(k,:),yh1(k,:,:),wo,xin(k,:),w1,yh2(k,:,:),w2);
    w11+=w3/n;
    w21+=w4/n;
    wo1+=wo5/n;
  endfor
  yh1=[];
  yh2=[];
  for k=1:n
    [ou,yh11,yh21,out1]=frwd(xin(k,:),xout(k,:),w1,w2,wo);
    err=[err;ou];
    yh1=[yh1;yh11];
    yh2=[yh2;yh21];
    out=[out;out1];
  endfor
errm=[errm,mean(err(:))];
id=[id,i];
endfor
plot(id,errm)
