package com.company.ChainOfResponsibility;

public abstract class Handler {
    //定义是那种级别
    public static final Integer FATHER_HANDLE_LEVEL = 1;
    public static final Integer HUSBAND_HANDLE_LEVEL = 2;
    public static final Integer SON_HANDLE_LEVEL = 3;

    //定义当前处于什么级别
    public int level;
    //下一个办理人是
    public Handler nextHandler;

    public Handler(int _level){
        this.level = _level;
    }

    public final void HandleMessage(Women women){
        if (women.getType() == level){
            this.response(women);
        } else {
            if (this.nextHandler != null){
                this.nextHandler.HandleMessage(women);
            }else{
                System.out.println("没人处理了");
            }

        }
    }
    /** 如果不属于你处理的请求，你应该让她找下一个环节的人，如女儿出嫁了，
     *   还向父亲请示是否可以逛街，那父亲就应该告诉女儿，应该找丈夫请示 */
    public void setNext(Handler _handler){ this.nextHandler = _handler; }
    //有请示那当然要回应
    protected abstract void response(IWomen women);
}
