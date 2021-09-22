package com.company.dymticproxy;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;

/**
 * 动态代理类
 */
public class GamePlayIH implements InvocationHandler {
    //被代理者
    Class cls = null;
    //被代理实例
    Object obj = null;
    //我要代理谁
    public GamePlayIH (Object _obj){
        this.obj = _obj;
    }
    //调用被代理的方法springAOP就在这里加上拦截器
    //静态代理和动态代理模式都是要求目标对象是实现一个接口的目标对象,但是有时候目标对象只是一个单独的对象,并没有实现任何的接口,
    // 这个时候就可以使用以目标对象子类的方式类实现代理,这种方法就叫做:Cglib代理
    @Override
    public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
        System.out.println("开启事务");
        Object result = method.invoke(this.obj, args);
        System.out.println("结束事务");
        //如果是登录方法，则发送信息
        if(method.getName().equalsIgnoreCase("login")){
            System.out.println("有人在用我的账号登录！");
        }
        return result;
    }
}
