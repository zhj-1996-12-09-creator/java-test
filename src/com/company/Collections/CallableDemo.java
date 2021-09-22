package com.company.Collections;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;

public class CallableDemo {

    public static void main(String[] args) throws ExecutionException, InterruptedException {
        Mythread mythread = new Mythread();
        FutureTask futureTask = new FutureTask(mythread);//适配类

        new Thread(futureTask,"A").start();
        new Thread(futureTask,"B").start(); // 第二次调用执行，会有结果缓存，不用 再次计算
        System.out.println(futureTask.get());


    }

    //函数式接口 function   输入啥返回啥
    static class Mythread implements Callable<Integer>{

        @Override
        public Integer call() throws Exception {
            System.out.println(Thread.currentThread().getName()+"=============>");
            TimeUnit.SECONDS.sleep(2);
            return 1024;
        }
    }
}
