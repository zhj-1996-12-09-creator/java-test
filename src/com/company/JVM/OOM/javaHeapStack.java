package com.company.JVM.OOM;

import sun.misc.VM;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

/**
 * java.lang.OutOfMemoryError: java Heap space
 */
public class javaHeapStack {


    private static void aVoid(){
        aVoid();
    }


    public static void main(String[] args) {
        /**
         * java.lang.OutOfMemoryError: stack over flow
         */
       // aVoid();

        /**
         * java.lang.OutOfMemoryError: java heap space 堆空间移出
         */
     //   Byte[] bytes = new Byte[30 * 1024 * 1024];


/**
 * CPU的98%时间都在做GC但是效果不佳，相当于没做
 * @param args   java.lang.OutOfMemoryError: GC overhead limit exceeded
 */
//Exception in thread "main"
       /*// Byte[] bytes = new Byte[30 * 1024 * 1024];
        int i = 0;
        List<String> list = new ArrayList<>();
        try {
            while (true){
                i++;
                list.add(String.valueOf(i).intern());
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("----------i:"+i);
            throw e;
        }*/

        /**
         * java.lang.OutOfMemoryError : Direct Buffer memory
         *
         * 直接内存溢出  直接内存就是Java堆内存以外的内存，
         * 像NIO中的ByteBuffer中
         *
         */
       /* System.out.println("直接内存---"+ VM.maxDirectMemory());//
        ByteBuffer.allocateDirect(8*1024*1024);//为对象分配的是本地内存。所以拷贝的速度快
        ByteBuffer.allocate(1);//这是在Java堆上分配*/

        /**
         * java.lang.OutOfMemory: unable to create new active thread
         * 不能创建本地线程了，一般时线程数多了，这和os有关
         * 要不改代码，减少线程数，要不修改服务器的配置
         */

        /**
         * MetaspaceSize  元空间主要存放
         * 虚拟机的类信息  rt.jar 下的class
         * 静态变量
         * 常量池
         * 即时编译后的代码
         */
    }

}
