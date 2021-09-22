package com.company;

import jdk.internal.org.objectweb.asm.tree.analysis.Value;

import java.nio.charset.StandardCharsets;
import java.time.*;
import java.util.*;
import java.util.stream.Stream;

public class java8<T> {
    public static< T > T defaultValue() {
        return null;
    }

    public T getOrDefault( T value, T defaultValue ) {
        return ( value != null ) ? value : defaultValue;
    }

    public static void main(String[] args) {
        /*Arrays.asList( "a", "b", "d" ).forEach(e -> System.out.println("简单是使用"+ e ) );
        Arrays.asList("f","h","a").forEach(e ->{
            if ("a".equals(e)){
                System.out.println("----找到我了--");
            }
        });*/
        //避免空指针
        /*Map map = new HashMap();
        map.put("1","1");
        Optional< Map > fullName = Optional.ofNullable( map );
        System.out.println( "Full Name is set? " + fullName.isPresent() );
        System.out.println( "Full Name: " + fullName.orElseGet( () -> null ) );
        System.out.println( fullName.map( a ->"Hey"+a.put("2","2")).orElse( "Hey Stranger!" ) );
        System.out.println(map);*/
// Get the system clock as UTC offset
        final Clock clock = Clock.systemUTC();
       /* System.out.println( clock.instant() );
        System.out.println( clock.millis() );
        System.out.println(System.currentTimeMillis());
        System.out.println(TimeZone.getDefault());*/

        // Get the local date and local time
        final LocalDate date = LocalDate.now();
        final LocalDate dateFromClock = LocalDate.now( clock );

        System.out.println( date );
        System.out.println( dateFromClock );

// Get the local date and local time
        final LocalTime time = LocalTime.now();
        final LocalTime timeFromClock = LocalTime.now( clock );

        System.out.println(time);
        System.out.println( timeFromClock );

        // Get the local date/time
        final LocalDateTime datetime = LocalDateTime.now();
        final LocalDateTime datetimeFromClock = LocalDateTime.now( clock );

        System.out.println( datetime );
        System.out.println( datetimeFromClock );

        // Get duration between two dates
        final LocalDateTime from = LocalDateTime.of( 2014, Month.APRIL, 16, 0, 0, 0 );
        final LocalDateTime to = LocalDateTime.of( 2015, Month.APRIL, 16, 23, 59, 59 );

        final Duration duration = Duration.between( from, to );
        System.out.println( "Duration in days: " + duration.toDays() );
        System.out.println( "Duration in hours: " + duration.toHours() );
//base64编码，不必再使用第三方类库
        final String text = "Base64 finally in Java 8!";

        final String encoded = Base64
                .getEncoder()
                .encodeToString( text.getBytes( StandardCharsets.UTF_8 ) );
        System.out.println( encoded );

        final String decoded = new String(
                Base64.getDecoder().decode( encoded ),
                StandardCharsets.UTF_8 );
        System.out.println( decoded );
        ArrayList list = new ArrayList();

          LinkedList list1 = new LinkedList();
    }
}
