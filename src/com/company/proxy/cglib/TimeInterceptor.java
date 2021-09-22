/*
package com.company.proxy.cglib;

import java.lang.reflect.Method;

*/
/**
 * MethodInterceptor和Enhancer
 * Enhancer 是字节码增强器
 * MethInterceptor 是需要自己实现
 *//*

public class TimeInterceptor implements MethodInterceptor {

    private Object target;
    public TimeInterceptor(Object target) {
        this.target = target;
    }
    @Override
    public Object intercept(Object proxy, Method method,
                            Object[] args, MethodProxy invocation) throws Throwable {
        System.out.println("方法之前：" + System.currentTimeMillis());
        Object ret = invocation.invoke(target, args);
        System.out.println("方法之后：" + System.currentTimeMillis());

        return ret;
    }

    class ProxyUtil {
        @SuppressWarnings("unchecked")
        public static <T> T proxyOne(Class<?> clz,MethodInterceptor interceptor){
            return (T)Enhancer.create(clz, interceptor);
        }

    }
}
*/
