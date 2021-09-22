/*
package com.company;

import com.company.proxy.DyProxy;
import com.company.proxy.DynamicProxy;
import com.company.proxy.MyInvocationHandler;
import com.company.proxy.Student;

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.lang.reflect.InvocationHandler;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;


public class User implements Externalizable {
    public transient String userName;


    @Override
    public void writeExternal(ObjectOutput out) throws IOException {

    }

    @Override
    public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {

    }
    static class thread1 implements Runnable{

        HashMap map = new HashMap<>();


        @Override
        public void run(){
           // synchronized (map){
                for(int i=0;i<25;i++){
                    map.put(i,i);
                }

           // }

        }
        public Integer getSize(){
            return map.size();
        }
        public static void main(String[] args) throws InterruptedException {
            Map map = new HashMap<>();
            Map map1 = new ConcurrentHashMap();
            Hashtable hashtable = new Hashtable();
            Map map2 = new TreeMap();
            Thread t1 = new Thread(new thread1());
            Thread t2 = new Thread(new thread1());
            t1.start();
            t2.start();

            System.out.println(new thread1().getSize());
        */
/*System.out.println(map.size());
        System.out.println(map1.size());
        System.out.println(hashtable.size());
        System.out.println(map2.size());*//*

        }

    }
    int uniquePaths(int m, int n){
        if (m == 1 || n == 1) {
            return 1;
        }

        int i, j;
         int [][]v = new int[m][n];
//二维数组里存的是根节点到他路径的条数
        for (i = 1; i < m; i++) {
            v[i][0] = 1;
            for (j = 1; j < n; j++) {
                v[0][j] = 1;
                v[i][j] = v[i][j-1] + v[i-1][j];
            }
        }

        return v[m-1][n-1];
    }
    public static void main(String[] args) {
        //定义一个主题
        DyProxy subject = new Student();
        // 定义一个Handler
         InvocationHandler handler = new MyInvocationHandler(subject);
        // 定义主题的代理
        DyProxy proxy = DynamicProxy.newProxyInstance(subject.getClass(). getClassLoader(), subject.getClass().getInterfaces(),handler);
        // 代理的行为
         proxy.getName();
    }
}
*/
