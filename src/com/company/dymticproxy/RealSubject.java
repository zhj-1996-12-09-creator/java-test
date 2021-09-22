package com.company.dymticproxy;

public class RealSubject implements Subject{
    @Override
    public void doSomething(String str) {

        System.out.println("do something!---->" + str);
        System.out.println("后置通知--");
    }

}
