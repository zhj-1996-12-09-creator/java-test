package com.company.Intermediary;

import java.util.Random;

public class Sale extends AbstractColleague{

    public Sale(AbstractMediator _mediator) {
        super(_mediator);
    }

    public void sellIBMComputer(int number){
        super.mediator.execute("sale.sell", number);
        System.out.println("销售IBM电脑"+number+"台");
    }

    //获得销售量
    public int getSaleStatus(){
        Random rand = new Random(System.currentTimeMillis());
        int saleStatus = rand.nextInt(100);
        System.out.println("Ibm电脑销售情况"+saleStatus);
        return saleStatus;
    }

    public void offSale(){ //库房有多少卖多少
        super.mediator.execute("sale.offsell");
    }
}
