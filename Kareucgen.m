kare=imread('kare.bmp');
daire=imread('daire.bmp');
ucgen=imread('ucgen.bmp');

karesize=imresize(kare,[100 100]);
dairesize=imresize(daire,[100 100]);
ucgensize=imresize(ucgen,[100 100]);

karedb=double(rgb2gray(karesize));
dairedb=double(rgb2gray(dairesize));
ucgendb=double(rgb2gray(ucgensize));

karers=reshape(karedb,10000,1);
dairers=reshape(dairedb,10000,1);
ucgenrs=reshape(ucgendb,10000,1);

E_Girisi=zeros(10000,3);
E_Girisi(:,1)=karers;
E_Girisi(:,2)=dairers;
E_Girisi(:,3)=ucgenrs;
Target=eye(3);

net=newff(minmax(E_Girisi),[20 20 3],{'logsig' 'logsig' 'logsig'},'trainscg');

net.trainParam.perf='sse';    
net.trainParam.epochs=7500;    
net.trainParam.goal=1e-100;    

net=train(net,E_Girisi,Target);

Testucgen=imread('Testucgen.bmp');
Testucgen2=imresize(Testucgen,[100 100]);
Testucgen3=double(rgb2gray(Testucgen2));
Testucgen4=reshape(Testucgen3,10000,1);
sim(net,Testucgen4)

Testdaire=imread('dairetest.bmp');
Testdaire2=imresize(Testdaire,[100 100]);
Testdaire3=double(rgb2gray(Testdaire2));
Testdaire4=reshape(Testdaire3,10000,1);
sim(net,Testdaire4)

Testkare=imread('Testkare.bmp');
Testkare2=imresize(Testkare,[100 100]);
Testkare3=double(Testkare2);
Testkare4=reshape(Testkare3,10000,1);
sim(net,Testkare4)
























