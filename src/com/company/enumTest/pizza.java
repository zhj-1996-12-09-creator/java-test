package com.company.enumTest;

import java.util.EnumMap;
import java.util.EnumSet;

public class pizza {
    public PizzaStatus getStatus() {
        return status;
    }

    public void setStatus(PizzaStatus status) {
        this.status = status;
    }

    private PizzaStatus status;
    private EnumSet<PizzaStatus> statuses = EnumSet.of(PizzaStatus.READY,PizzaStatus.DELIVERED,PizzaStatus.ORDERED);
    private EnumMap<PizzaStatus,pizza> enumMap = null;
    public enum PizzaStatus {
        ORDERED (5){
            @Override
            public boolean isOrdered() {
                System.out.println("999999");
                return true;
            }
        },
        READY (2){
            @Override
            public boolean isReady() {
                return true;
            }
        },
        DELIVERED (0){
            @Override
            public boolean isDelivered() {
                return true;
            }
        };

        private int timeToDelivery;

        public boolean isOrdered() {return false;}

        public boolean isReady() {return false;}

        public boolean isDelivered(){return false;}

        public int getTimeToDelivery() {
            return timeToDelivery;
        }

        PizzaStatus (int timeToDelivery) {
            this.timeToDelivery = timeToDelivery;
        }
    }

    public boolean isDeliverable() {
        return this.status.isReady();
    }

    public void printTimeToDeliver() {
        System.out.println("Time to delivery is " +
                this.getStatus().getTimeToDelivery());
    }


    public static void main(String[] args) {
        pizza testPz = new pizza();
        testPz.setStatus(PizzaStatus.READY);
        System.out.println(testPz.isDeliverable());
    }
}
