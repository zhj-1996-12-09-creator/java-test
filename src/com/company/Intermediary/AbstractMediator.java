package com.company.Intermediary;

/**
 * 利用中介者模块整合其他模块，其他模块想要交流就得利用中介者模块
 */

public abstract class AbstractMediator {
    protected Purchase purchase;
    protected Sale sale;
    protected Stock stock;

    public AbstractMediator(){
        purchase = new Purchase(this);
        sale = new Sale(this);
        stock = new Stock(this);
    }

    public abstract void execute(String str,Object...objects);
}
