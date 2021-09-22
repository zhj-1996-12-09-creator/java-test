
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;

/**
 * @author 张海杰
 */
public class TreadTest implements Runnable {

    private int b;

    //不能少static，可以对象初始化时会没有完全初始化好
    private static final ReentrantLock reentrantLock = new ReentrantLock();
    private  AtomicInteger atomicInteger = null;



    public TreadTest(int b) {
        this.b = b;
        this.atomicInteger = new AtomicInteger(b);
    }

    public void setB(int b){
        this.b= b;
    }
    //线程的状态：新增,就绪，运行，阻塞，终止，

    // synchronized 代码块
    //notify()随机唤醒是的意思是 根据JDK版本不同在等待队列中唤醒的线程
    //在队列里面的位置不同 所以被称为 ：“选择是任意性的” JDK1.8-中调用notify()
    //唤醒的是等待队列中的头节点（等待时间最长的那个线程）

    /**
     * 可重入锁
     * 一个线程在持有一个锁的时候，它内部能否再次（多次）申请该锁。
     * 如果一个线程已经获得了锁，其内部还可以多次申请该锁成功。那么我们就称该锁为可重入锁
     * <p>
     * 一个最主要的就是ReentrantLock还可以实现公平锁机制。什么叫公平锁呢？
     * 也就是在锁上等待时间最长的线程将获得锁的使用权。通俗的理解就是谁排队时间最长谁先执行获取锁
     * <p>
     * java 的锁：独占锁和共享锁
     * 独占锁：是指该锁一次只能被一个线程持有
     * 共享锁: 可以被多个线程持有，一般读锁 ReentrantReadWriteLock，读写锁，读锁是共享锁，写锁是独占锁
     * synchronized 是由一对 monitorenter/monitorexit 指令实现的
     * ThreadLocal为每个线程创建了一个副本变量，这个ThreadLocalMap只有当前线程可以访问，具有线程隔离性
     * createMap(t, value);所以set会有覆盖情况
     */

    /*@Override
    public void run() {
        //果然线程安全了 synchronized  1621927949101  1621927949106
        //   reentrantLock           1621928062586  1621928062590    ---效率高
        System.out.println("第"+b+"个线程执行开始时间"+System.currentTimeMillis());
       *//* synchronized (TreadTest.class) {
            for (int i = 0; i < 1000000; i++) {

                //System.out.println("第" + b + "个线程执行l  --" + i);
            }
        }*//*



        ThreadLocal threadLocal = new ThreadLocal();
        if(b==1){
            threadLocal.set("A");
            threadLocal.set("hello");
        }
        System.out.println(threadLocal.get());
        reentrantLock.lock();
        try{
            for (int i = 0; i < 1000000; i++) {
                *//*System.out.println("第" + b + "个线程执行l  --"+i);
                if(i==25&&b==3){
                    //只是唤醒，重新加入队列
                    TreadTest.class.notify();
                }
                if (i==50){
                    try {
                        //释放锁：sleep() 不释放锁；wait() 释放锁。
                        //wait是Object的方法
                        //sleep会自动恢复
                        //Thread.sloop(1);
                        TreadTest.class.wait();
                        System.out.println("睡着了");
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }*//*
 }
        }catch (Exception e){

        }finally {
            reentrantLock.unlock();
            System.out.println("第"+b+"个线程执行结束时间"+System.currentTimeMillis());
        }

    }*/

    @Override
    public void run() {
        this.atomicInteger.getAndAdd(9);
    }
    public static void main(String[] args) {
        TreadTest treadTest1 = new TreadTest(1);
        TreadTest treadTest2 = new TreadTest(1);
        TreadTest treadTest3 = new TreadTest(1);
        //线程交替执行，时间片轮转
        Thread t1 = new Thread(treadTest1);
        t1.start();

        new Thread(treadTest2).start();
        new Thread(treadTest3).start();


    }


}
