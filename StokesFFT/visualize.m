clear;
clc;

load("u.csv")
load("v.csv")
load("p.csv")
load("x.csv")
load("y.csv")
load("u_acc.csv")

figure(1)
contourf(x,y,u,50,'linestyle',"none")
colormap(jet)
daspect([1 1 1]);
disp("u deal")
colorbar
xlabel("u")

figure(2)
contourf(x,y,v,50,'linestyle',"none")
colormap(jet)
daspect([1 1 1]);
disp("v deal")
colorbar
xlabel("v")

figure(3)
contourf(x,y,p,50,'linestyle',"none")
colormap(jet)
daspect([1 1 1]);
disp("p deal")
colorbar
xlabel("p")

figure(4)
acc = 1/16*sin(4*y);
contourf(x,y,acc,50,'linestyle',"none")
colormap(jet)
daspect([1 1 1]);
disp("u_acc deal");
colorbar;
xlabel("u_{acc}");

error2 = sum(sum((u-acc).*(u-acc)));
figure(5)
contourf(x,y,u-acc,50,'linestyle','none')
colormap(jet);
daspect([1 1 1]);
disp("error deal")
colorbar
title("error L^2:"+num2str(error2));
xlabel("u-u_{acc}");
