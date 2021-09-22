package com.company.proxy;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Proxy;

/**
 * 动态代理是指在不改变代码的前提下实现对原有方法的增强
 * 切入点：需要增强的接口方法
 * 通知：增强的内容
 * 连接点：接口中所有方法、包括不增强的方法
 * 织入：将通知放入切点
 * 切面：切点和通知
 * @param <T>
 */
public class DynamicProxy<T> {

    public static <T> T newProxyInstance(ClassLoader loader, Class<?>[] interfaces, InvocationHandler h){
        //寻找JoinPoint连接点，AOP框架使用元数据定义
        if(true){//执行一个前置通知
            System.out.println("前置通知");
        }
        // 执行目标，并返回结果
         return (T) Proxy.newProxyInstance(loader,interfaces, h); }
}
