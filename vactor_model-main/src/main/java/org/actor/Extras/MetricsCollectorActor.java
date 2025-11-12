package org.actor.Extras;

import akka.actor.typed.ActorRef;
import akka.actor.typed.Behavior;
import akka.actor.typed.javadsl.AbstractBehavior;
import akka.actor.typed.javadsl.ActorContext;
import akka.actor.typed.javadsl.Behaviors;
import akka.actor.typed.javadsl.Receive;
import org.actor.CoordinatorActor;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class MetricsCollectorActor extends AbstractBehavior<MetricsCollectorActor.Command> {

    public interface Command {}

    public static class LatencyEvent implements Command {
        public final String actorName;
        public final String messageType;
        public final long latencyMicros;
        public LatencyEvent(String actorName, String messageType, long latencyMicros) {
            this.actorName = actorName;
            this.messageType = messageType;
            this.latencyMicros = latencyMicros;
        }
    }

    public static class LoadEvent implements Command {
        public final String actorName;
        public final long count;
        public LoadEvent(String actorName, long count) {
            this.actorName = actorName;
            this.count = 1;
        }
    }

    private BufferedWriter logWriter;
    final ActorRef<CoordinatorActor.Command> parent;
    private final String logFilePath;

    public static Behavior<Command> create(ActorRef<CoordinatorActor.Command> parent, String logFile) {
        return Behaviors.setup(ctx -> new MetricsCollectorActor(ctx, parent, logFile));
    }

    public MetricsCollectorActor(ActorContext<Command> context, ActorRef<CoordinatorActor.Command> parent, String logFile)
            throws IOException {
        super(context);
        this.parent = parent;
        this.logFilePath = logFile;

        File file = new File(logFile);
        boolean isNewFile = !file.exists();
        this.logWriter = new BufferedWriter(new FileWriter(file, true));

        if (isNewFile) {
            // Write CSV header
            logWriter.write("type,actor_name,message_type_or_count,value\n");
            logWriter.flush();
        }
    }

    @Override
    public Receive<Command> createReceive() {
        return newReceiveBuilder()
                .onMessage(LatencyEvent.class, this::onLatency)
                .onMessage(LoadEvent.class, this::onLoad)
                .build();
    }

    private Behavior<Command> onLatency(LatencyEvent evt) {
        try {
            logWriter.write(String.format("LATENCY,%s,%s,%d\n", evt.actorName, evt.messageType, evt.latencyMicros));
            logWriter.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return this;
    }

    private Behavior<Command> onLoad(LoadEvent evt) {
        try {
            // Using message_type_or_count column for consistency
            logWriter.write(String.format("LOAD,%s,COUNT,%d\n", evt.actorName, evt.count));
            logWriter.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return this;
    }
}
