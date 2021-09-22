package com.company.dymticproxy;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Proxy;

public class clent {
    public static void main(String[] args) {
        GamePlayer player = new GamePlayer("张三");
      // GamePlayerProxy proxy = new GamePlayerProxy(player);
        //定义一个handler
        InvocationHandler handler = new GamePlayIH(player);
        //获得类的class loader
        ClassLoader cl = player.getClass().getClassLoader();
        //动态产生一个代理者 需要接口
        IGamePlayer proxy = (IGamePlayer) Proxy.newProxyInstance(cl,new Class[]{IGamePlayer.class},handler);
        proxy.login("张三","123");
        proxy.killBoss();
        proxy.upgrade();
    }

    /*public static void main(String[] args) {
        //定义一个主题
        Subject subject = new RealSubject();
        Subject proxy = SubjectDynamicProxy.newProxyInstance(subject);
        *//*//*///定义一个Handler
       // InvocationHandler handler = new MyInvocationHandler(subject);
        //定义主题的代理
      //  Subject proxy = DynamicProxy.newProxyInstance(subject.getClass(). getClassLoader(), subject.getClass().getInterfaces(),handler);*//*
        //代理的行为
      //  proxy.doSomething("Finish");
   // }*/
}
