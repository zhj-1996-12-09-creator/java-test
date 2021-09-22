package com.company.proxy;

import java.util.*;

public class Student implements DyProxy {
    @Override
    public void getName() {
        System.out.println("张三！！！");
    }

    @Override
    public void getSex() {

    }

    @Override
    public void getAge() {

    }

    public static void main(String[] args) {
       /* List<Integer> classList=new ArrayList<Integer>();
        classList.add(1);
        classList.add(1);
        classList.add(2);
        classList.add(3);
        HashSet<Integer> classSet=new HashSet<Integer>();
        classSet.addAll(classList);     //addAll()可以复制其他集合的元素，效果就跟一个一个加进去一样
        System.out.println(classSet.size());*/
        Student student = new Student();
        MyInvocationHandler myInvocationHandler = new MyInvocationHandler(student);
        DynamicProxy.newProxyInstance(student.getClass().getClassLoader(),student.getClass().getInterfaces(),myInvocationHandler);
        student.getName();
        Hashtable<String,Object> ta = new Hashtable();
        TreeMap<String,Object> jj = new TreeMap<>();
    }


}
