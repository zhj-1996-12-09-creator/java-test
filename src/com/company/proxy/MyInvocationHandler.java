package com.company.proxy;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;

public class MyInvocationHandler implements InvocationHandler {

    private Object target = null ;

    public MyInvocationHandler(DyProxy subject) {
        this.target = subject;
    }


   /* public Object MyInvocationHandler(Object ta){
        this.target = ta;
    }*/

    @Override
    public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
        return method.invoke(target);
    }
}